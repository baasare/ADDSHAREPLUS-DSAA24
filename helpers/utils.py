import json
import os
import signal
import random
import pickle
import codecs
import socket
import logging
import requests
import itertools
import numpy as np
import pandas as pd
from PIL import Image
import scipy.io as sio
import tensorflow as tf
from timeit import default_timer
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.asymmetric import rsa, ec
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import serialization, hashes


def post_with_retries(url, data, max_retries=3):
    session = requests.Session()
    retries = Retry(total=max_retries, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    try:
        response = session.post(url, json=data, headers={"Content-Type": "application/json"})
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None


def decode_layer(b64_str):
    return pickle.loads(codecs.decode(b64_str.encode(), "base64"))


def encode_layer(layer):
    return codecs.encode(pickle.dumps(layer), "base64").decode()


def check_port(address, port):
    # Check if the port is in use
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((address, port))
    except socket.error:
        return True
    finally:
        sock.close()
    return False


def terminate_process_on_port(port):
    try:
        pid = int(os.popen(f"lsof -t -i:{port}").read())
        os.kill(pid, signal.SIGKILL)
        print(f"Terminated process on port {port}")
    except Exception as e:
        print(f"Error terminating process on port {port}: {e}")


def fetch_index(dataset):
    current_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    path = current_dir + f'/resources/dataset/{dataset}/iid_balanced.txt'
    clients = np.loadtxt(path, dtype=object)
    clients = clients.astype(np.float64)
    clients = clients.astype(np.int64)
    return clients


def fetch_dataset(dataset):
    if dataset == "cifar-10":
        return tf.keras.datasets.cifar10.load_data()
    elif dataset == "mnist":
        return tf.keras.datasets.mnist.load_data()
    elif dataset == "f-mnist":
        return tf.keras.datasets.fashion_mnist.load_data()
    else:
        current_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        train_data = sio.loadmat(current_dir + f'/resources/dataset/svhn/train_32x32.mat')
        test_data = sio.loadmat(current_dir + f'/resources/dataset/svhn/test_32x32.mat')
        x_train = np.array(train_data['X'])
        y_train = np.array(train_data['y'])

        x_test = np.array(test_data['X'])
        y_test = np.array(test_data['y'])

        return (x_train, y_train), (x_test, y_test)


def get_dataset(index, dataset, x_train, y_train, x_test, y_test):
    if dataset == "cifar-10":
        x_train = x_train[index]
        y_train = y_train[index]

        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0

        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)

    elif dataset == 'svhn':
        x_train = np.moveaxis(x_train, -1, 0)
        x_test = np.moveaxis(x_test, -1, 0)

        x_train = x_train[index]
        y_train = y_train[index]

        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0

        y_train = tf.keras.utils.to_categorical((y_train - 1), 10)
        y_test = tf.keras.utils.to_categorical((y_test - 1), 10)

    else:
        x_train = x_train[index]
        y_train = y_train[index]

        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255.0
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255.0

        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test


def get_lenet5_classification(dataset):
    """
    Creates LeNet-5 model.

    :param dataset: dataset being used for model
    :return: Keras model
    """

    if dataset == "cifar-10" or dataset == 'svhn':
        pixel = 32
        rgb = 3
    else:
        pixel = 28
        rgb = 1

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # LeNet-5
    model = tf.keras.models.Sequential()
    # Convolutional layer 1
    model.add(tf.keras.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        strides=1,
        activation='relu',
        name=f'conv2d_{0}',
        input_shape=(pixel, pixel, rgb)
    ))
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Convolutional layer 2
    model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu', name=f'conv2d_{1}'))
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Flatten the output of the convolutional layers
    model.add(tf.keras.layers.Flatten())

    # Fully connected layer 1
    model.add(tf.keras.layers.Dense(units=120, activation='relu', name=f'dense_{0}'))

    # Fully connected layer 2
    model.add(tf.keras.layers.Dense(units=84, activation='relu', name=f'dense_{1}'))

    # Output layer
    model.add(tf.keras.layers.Dense(units=10, activation='softmax', name=f'dense_{2}'))
    return model


def get_lenet5_regression():
    """
    Creates LeNet-5 model.

    :return: Keras model
    """

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # LeNet-5
    model = tf.keras.models.Sequential()
    # Convolutional layer 1
    model.add(tf.keras.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        strides=1,
        activation='relu',
        name=f'conv2d_{0}',
        input_shape=(512, 512, 3)
    ))
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Convolutional layer 2
    model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu', name=f'conv2d_{1}'))
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Flatten the output of the convolutional layers
    model.add(tf.keras.layers.Flatten())

    # Fully connected layer 1
    model.add(tf.keras.layers.Dense(units=120, activation='relu', name=f'dense_{0}'))

    # Fully connected layer 2
    model.add(tf.keras.layers.Dense(units=84, activation='relu', name=f'dense_{1}'))

    # Output layer
    model.add(tf.keras.layers.Dense(units=1, activation='softmax', name=f'dense_{2}'))
    return model


def get_regression_model():
    """
    :return: Keras model
    """

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation='relu',
        name=f'conv2d_{0}',
        input_shape=(512, 512, 3))
    )
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), activation='relu', name=f'conv2d_{1}'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=128, activation='relu', name=f'dense_{0}'))
    model.add(tf.keras.layers.Dense(units=32, activation='relu', name=f'dense_{1}'))
    model.add(tf.keras.layers.Dense(units=1, name=f'dense_{2}'))

    return model


def get_lenet3(dataset):
    """
    Creates LeNet-5 model.

    :param dataset: dataset being used for model
    :return: Keras model
    """

    if dataset == "cifar-10" or dataset == 'svhn':
        pixel = 32
        rgb = 3
    else:
        pixel = 28
        rgb = 1

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # LeNet-5
    model = tf.keras.models.Sequential()
    # Convolutional layer 1
    model.add(tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), strides=1, activation='relu',
                                     input_shape=(pixel, pixel, rgb)))
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Flatten the output of the convolutional layers
    model.add(tf.keras.layers.Flatten())

    # Fully connected layer 1
    model.add(tf.keras.layers.Dense(units=120, activation='relu'))

    # Output layer
    model.add(tf.keras.layers.Dense(units=10, activation='softmax'))
    return model


def generate_keys(save_path, name, encryption_type, nbits=4096):
    """
    :param save_path:
    :param nbits:
    :param name:
    :param encryption_type:
    :return:
    """
    os.makedirs(save_path, exist_ok=True)
    private_key, public_key = None, None

    if encryption_type == 'rsa':
        # Generate an RSA key pair
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=nbits
        )
        public_key = private_key.public_key()
    else:
        # Generate recipient key pair
        print("Elliptical")
        private_key = ec.generate_private_key(ec.SECP256R1(), default_backend())
        public_key = private_key.public_key()

    public_pem_path = os.path.join(save_path, 'client_' + str(name) + '_public.pem')
    private_pem_path = os.path.join(save_path, 'client_' + str(name) + '_private.pem')

    try:
        # Serialize the private key to PEM format and save it to a file
        with open(private_pem_path, 'wb') as f:
            f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))

        # Serialize the public key to PEM format and save it to a file
        with open(public_pem_path, 'wb') as f:
            f.write(public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            ))
    except Exception as ex:
        logging.error(ex)

    return public_pem_path, private_pem_path


def encrypt_message_elliptical(message, recipient_public_key):
    ephemeral_private_key = ec.generate_private_key(ec.SECP256R1(), default_backend())
    ephemeral_public_key = ephemeral_private_key.public_key()

    shared_secret = ephemeral_private_key.exchange(ec.ECDH(), recipient_public_key)

    derived_key_material = HKDF(
        algorithm=hashes.SHA256(),
        length=32 + 12,
        salt=None,
        info=b'',
    ).derive(shared_secret)

    encryption_key = derived_key_material[:32]
    nonce = derived_key_material[32:]

    cipher = AESGCM(encryption_key)
    ciphertext = cipher.encrypt(nonce, message.encode(), None)

    return ephemeral_public_key, ciphertext


def decrypt_message_elliptical(ciphertext, ephemeral_public_key, recipient_private_key):
    shared_secret = recipient_private_key.exchange(ec.ECDH(), ephemeral_public_key)

    derived_key_material = HKDF(
        algorithm=hashes.SHA256(),
        length=32 + 12,
        salt=None,
        info=b'',
    ).derive(shared_secret)

    encryption_key = derived_key_material[:32]
    nonce = derived_key_material[32:]

    cipher = AESGCM(encryption_key)
    decrypted_message = cipher.decrypt(nonce, ciphertext, None)

    return decrypted_message.decode()


def iid_balanced(client_number, train_size, dataset):
    # used to generate indexes
    rand_array = np.arange(train_size)
    np.random.shuffle(rand_array)

    clients = [[] for _ in range(client_number)]

    for i in range(client_number):
        clients[i] = rand_array[
                     int(i * train_size / client_number):int((i + 1) * train_size / client_number)]

    np.savetxt(f"resources/dataset/{dataset}/iid_balanced.txt", clients)


def generate_groups(nodes, group_size):
    # shuffle clients
    random.shuffle(nodes)

    # Create a repeating iterator over the clients list
    client_iterator = itertools.cycle(nodes)

    # Keep track of which clients have been selected
    selected_clients = set()

    # Keep selecting groups of clients until each client has been selected at least once
    selected_groups = []
    while len(selected_clients) < len(nodes):
        # Select the next group of clients
        group = list(itertools.islice(client_iterator, group_size))

        # Add the selected clients to the set of selected clients
        selected_clients.update(group)

        # Add the selected group to the list of selected groups
        selected_groups.append(group)

    # All clients have been selected at least once
    return selected_groups


def get_public_key(client_id, encryption_type='rsa'):
    current_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    path = current_dir + f'/resources/keys/{encryption_type}/client_{str(client_id)}_public.pem'

    with open(path, 'rb') as f:
        return serialization.load_pem_public_key(f.read())


def get_private_key(client_id, encryption_type='rsa'):
    current_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    path = current_dir + f'/resources/keys/{encryption_type}/client_{str(client_id)}_private.pem'

    with open(path, 'rb') as f:
        return serialization.load_pem_private_key(f.read(), password=None)


class TimingCallback(tf.keras.callbacks.Callback):
    def __init__(self, logs=None):
        super().__init__()
        if logs is None:
            logs = {}
        self.start_time = None
        self.logs = []

    def on_epoch_begin(self, epoch, logs=None):
        if logs is None:
            logs = {}
        self.start_time = default_timer()

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        self.logs.append(default_timer() - self.start_time)


def convert(imgfile, labelfile, outfile, n):
    f = open(imgfile, "rb")
    o = open(outfile, "w")
    l = open(labelfile, "rb")

    f.read(16)
    l.read(8)
    images = []

    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28 * 28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image) + "\n")
    f.close()
    o.close()
    l.close()


def f_to_i(x, scale=1 << 32):
    if x < 0:
        if pow(2, 64) - (abs(x) * scale) > (pow(2, 64) - 1):
            return np.uint64(0)
        x = pow(2, 64) - np.uint64(abs(x) * scale)
    else:
        x = np.uint64(scale * x)
    return np.uint64(x)


def i_to_f(x, scale=1 << 32):
    l = 64
    t = x > (pow(2, (l - 1)) - 1)
    if t:
        x = pow(2, l) - x
        y = np.uint64(x)
        y = np.float32(y * (-1)) / scale
    else:
        y = np.float32(np.uint64(x)) / scale
    return y


def generate_integer_additive_shares(value, n):
    arr = np.asarray(value)
    rand_arr = np.random.randint(1000, size=(n - 1,) + arr.shape)
    shares = np.concatenate((rand_arr, [arr - rand_arr.sum(axis=0)]), axis=0)
    return shares


def generate_additive_shares(value, n):
    arr = np.asarray(value)
    rand_arr = np.random.uniform(low=-np.abs(arr), high=np.abs(arr), size=(n - 1,) + arr.shape)
    shares = np.concatenate((rand_arr, [arr - rand_arr.sum(axis=0)]), axis=0)
    return shares


def random_weight_selection(weights, fraction):
    percentage = max(0, min(100, fraction))
    flattened_weights = weights.flatten()
    num_elements = int(np.ceil(percentage * flattened_weights.size))
    indexes = np.random.choice(flattened_weights.size, size=num_elements, replace=False)
    original_indices = np.unravel_index(indexes, weights.shape)
    indices = [arr.tolist() for arr in original_indices]
    return indices


def magnitude_weight_selection(weights, fraction):
    percentage = max(0, min(100, fraction))
    num_elements = int(np.ceil(percentage / 100 * weights.size))
    indices_of_largest = np.argpartition(weights.flatten(), -num_elements)[-num_elements:]
    original_indices = np.unravel_index(indices_of_largest, weights.shape)
    indices = [arr.tolist() for arr in original_indices]
    return indices


def obd_weight_selection(model, x, y, weights, fraction):
    with tf.GradientTape() as tape:
        predictions = model(x)
        loss = tf.keras.losses.categorical_crossentropy(y, predictions)
        gradients = tape.gradient(loss, weights)
        sensitivities = [grad * weight for grad, weight in zip(gradients, weights)]
        sensitivities = np.array(sensitivities)
        return magnitude_weight_selection(sensitivities, fraction)


def regularization_weight_selection(model, x, y, reg_type, weights, fraction):
    regularization_lambda = 0.01
    l1_regularization = regularization_lambda * tf.reduce_sum(tf.abs(weights))
    l2_regularization = regularization_lambda * tf.reduce_sum(tf.square(weights))

    with tf.GradientTape() as tape:
        predictions = model(x)
        loss = tf.keras.losses.categorical_crossentropy(y, predictions)
        gradients = tape.gradient(loss, weights)

    total_gradient = gradients + (l1_regularization if reg_type == "l1" else l2_regularization)
    l1_weight = np.array(weights - total_gradient)
    return magnitude_weight_selection(l1_weight, fraction)


def combine_csv_files(experiment, dataset):
    parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    folder_path = parent_dir + f'/resources/results/{experiment}/{dataset}'

    # List all files in the folder
    files = os.listdir(folder_path)

    # Initialize an empty list to store the data frames
    data_frames = []

    # Iterate over each file
    for file in files:
        if file.startswith('client') and file.endswith('.csv'):  # Ensure that only CSV files are considered
            file_path = os.path.join(folder_path, file)

            # Read the CSV file and append it to the list
            df = pd.read_csv(file_path)
            df = df.drop(columns='round')
            data_frames.append(df)

    # Concatenate all data frames into one
    if len(data_frames) != 0:
        combined_df = pd.concat(data_frames, axis=1)
        combined_df.to_csv(f"{folder_path}/combined.csv")

    for file in files:
        if file.startswith('client') and file.endswith('.csv'):
            file_path = os.path.join(folder_path, file)
            os.remove(file_path)

    # return combined_df


def combine_find_mean(experiment, dataset):
    parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    folder_path = parent_dir + f'/resources/results/{experiment}/{dataset}'

    combine_csv_files(experiment, dataset)

    csv_dir = os.path.join(folder_path, 'combined.csv')

    if os.path.exists(csv_dir):
        df = pd.read_csv(csv_dir)
        df['Average Accuracy'] = df.apply(lambda row: row['accuracy'].mean(), axis=1)
        df['Average Training'] = df.apply(lambda row: row['training'].mean(), axis=1)
        if 'secret_sharing' in df:
            df['Average Secret Sharing'] = df.apply(lambda row: row['secret_sharing'].mean(), axis=1)
        df.columns = df.columns.str.replace(r'\.\d+$', '', regex=True)
        df.to_csv(csv_dir, index=False)


def combine_find_mean_regression(experiment, dataset):
    parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    folder_path = parent_dir + f'/resources/results/{experiment}/{dataset}'

    # combine_csv_files(experiment, dataset)

    csv_dir = os.path.join(folder_path, 'combined.csv')

    if os.path.exists(csv_dir):
        df = pd.read_csv(csv_dir)
        df['Average Loss'] = df.apply(lambda row: row['loss'].mean(), axis=1)
        df['Average RMSE'] = df.apply(lambda row: row['rmse'].mean(), axis=1)
        df['Average MAPE'] = df.apply(lambda row: row['mape'].mean(), axis=1)
        df['Average Training'] = df.apply(lambda row: row['training'].mean(), axis=1)
        if 'secret_sharing' in df:
            df['Average Secret Sharing'] = df.apply(lambda row: row['secret_sharing'].mean(), axis=1)
        df.columns = df.columns.str.replace(r'\.\d+$', '', regex=True)
        df.to_csv(csv_dir, index=False)


def convert_png_to_eps(folder_path):
    files = os.listdir(folder_path)

    for file in files:
        if file.startswith('multiline') and file.endswith('.png'):  # Ensure that only png files are considered
            # file name
            png_file = os.path.join(folder_path, file)

            # Open the PNG file
            image = Image.open(png_file)

            # Create a new EPS file with the same dimensions and mode as the PNG
            eps_image = Image.new("RGB", image.size)
            eps_image.paste(image)

            # Save the EPS file
            eps_image.save(png_file.replace("png", "eps"), format='EPS')


def combine_find_mean_2():
    os.system('find . -name ".DS_Store" -delete')
    parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    folder_path = parent_dir + f'/resources/results'
    datasets = ['cifar-10', 'f-mnist', 'mnist', 'svhn']
    folder_items = os.listdir(folder_path)

    for folder in folder_items:
        for dataset in datasets:
            combine_csv_files(folder, dataset)

            csv_dir = os.path.join(folder_path, folder, dataset, 'combined.csv')

            if os.path.exists(csv_dir):
                df = pd.read_csv(csv_dir)
                df['Average Accuracy'] = df.apply(lambda row: row['accuracy'].mean(), axis=1)
                df['Average Training'] = df.apply(lambda row: row['training'].mean(), axis=1)
                if 'secret_sharing' in df:
                    df['Average Secret Sharing'] = df.apply(lambda row: row['secret_sharing'].mean(), axis=1)
                df.columns = df.columns.str.replace(r'\.\d+$', '', regex=True)
                df.to_csv(csv_dir, index=False)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, tf.Variable):
            return self.default(obj.numpy())  # Convert tf.Variable to numpy array
        return json.JSONEncoder.default(self, obj)


class NumpyDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.dict_to_object, *args, **kwargs)

    def dict_to_object(self, d):
        if isinstance(d, dict):
            for key, value in d.items():
                if isinstance(value, list):
                    for i, v in enumerate(value):
                        if isinstance(v, list):
                            value[i] = self.list_to_ndarray(v)
                d[key] = value
        return d

    def list_to_ndarray(self, l):
        for i, v in enumerate(l):
            if isinstance(v, list):
                l[i] = self.list_to_ndarray(v)
            elif isinstance(v, str):
                try:
                    l[i] = np.fromstring(v[1:-1], sep=',')
                except:
                    pass
        return np.array(l, dtype=object)


def resize_images(source_path, destination_path, target_size=(512, 512)):
    """Resizes images in a source directory to a target size and saves them in a destination directory.

    Args:
        source_path (str): Path to the directory containing the original images.
        destination_path (str): Path to the directory where the resized images will be saved.
        target_size (tuple, optional): Desired size of the resized images. Defaults to (224, 224).
    """

    os.makedirs(destination_path, exist_ok=True)

    for filename in os.listdir(source_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            source_file = os.path.join(source_path, filename)
            dest_file = os.path.join(destination_path, filename)

            try:
                with Image.open(source_file) as image:
                    resized_image = image.resize(target_size)
                    resized_image.save(dest_file)

            except FileNotFoundError:
                print(f"Error: Image file not found at {source_file}")
            except Exception as e:
                print(f"Error resizing image {filename}: {e}")


def create_and_save_dataset(field, dataset_path='dataset.npz', image_size=(512, 512)):
    current_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    yield_data = pd.read_csv(os.path.join(current_dir, f'resources/dataset/area_x/field{field}/merged.csv'))
    df = pd.DataFrame(
        {
            'file_path': yield_data['train_file_path'].tolist(),
            'label': yield_data['yield'].values
        }
    )

    images = []
    labels = []

    for _, row in df.iterrows():
        file_path = row['file_path']
        label = row['label']

        img = tf.keras.preprocessing.image.load_img(file_path, target_size=image_size)
        img = tf.keras.preprocessing.image.img_to_array(img) / 255.0

        images.append(img)
        labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    np.savez_compressed(dataset_path, images=images, labels=labels)


def load_area_x_dataset(dataset_path='dataset.npz'):
    dataset = np.load(dataset_path)
    images = dataset['images']
    labels = dataset['labels']
    return images, labels


def get_area_x_dataset(field):
    current_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    yield_data_file = os.path.join(current_dir, f'resources/dataset/area_x/field{field}/merged.csv')
    yield_data = pd.read_csv(yield_data_file)

    labels = yield_data['yield'].values

    x_train = np.array([load_and_preprocess_image(path, (512, 512)) for path in yield_data['train_file_path'].tolist()])
    x_test = np.array([load_and_preprocess_image(path, (512, 512)) for path in yield_data['test_file_path'].tolist()])
    y_train, y_test = labels, labels

    return x_train, y_train, x_test, y_test


def load_and_preprocess_image(image_path, target_size):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    return img_array


if __name__ == "__main__":
    # combine_find_mean_2()
    # combine_find_mean("addshare_encrypted", "mnist")

    import pandas as pd

    parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    folder_path = parent_dir + f'/resources/results/addshare_encrypted/mnist/'
    df = pd.read_csv(folder_path + 'combined.csv')
    accuracy_mean = df['accuracy'].mean()
    training_mean = df['training'].mean()
    secret_sharing_mean = df['secret_sharing'].mean()

    # Print the results
    print(f"Average accuracy: {accuracy_mean:.6f}")
    print(f"Average training: {training_mean:.6f}")
    print(f"Average secret_sharing: {secret_sharing_mean:.6f}")

# for field in ["1"]:
#     month = "august"
#     data_type = "test"
#     dataset_path = keys_path = os.path.join(os.path.dirname(os.getcwd()), 'resources', 'dataset', 'area_x')
#     yield_data_file = dataset_path + f"/field{field}/merged.csv"
#     input_file_path = dataset_path + f"/field{field}/{month}.tif"
#     output_file_path = dataset_path + f"/field{field}/images/{month}"
#     crop_tiles_from_geotiff(input_file_path, output_file_path, yield_data_file, data_type)
#
#     images_path = dataset_path + f"/field{field}/images/{month}"
#     resize_images(images_path, images_path)
#
#     dataset = dataset_path + f"/field{field}/{data_type}.npz"
#     create_and_save_dataset(field, dataset_path=dataset)
#     keys_path = os.path.join(os.path.dirname(os.getcwd()), 'resources', 'keys', 'elliptical')
#     # generate encryption keys for all clients
#     for i in range(50):
#         generate_keys(keys_path, 1 + i, 'elliptical')
