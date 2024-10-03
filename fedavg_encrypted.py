import os
import sys
import time
import json
import uvicorn
import threading
import pandas as pd
import tensorflow as tf
from server import Server
from fastapi import FastAPI
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding

from helpers.utils import get_public_key, TimingCallback, NumpyEncoder
from helpers.utils import check_port, terminate_process_on_port, decode_layer
from helpers.utils import fetch_dataset, fetch_index, get_dataset, post_with_retries, encode_layer

from helpers.constants import CLIENT_PORT, SERVER_ID, NODES, MESSAGE_END_SESSION, EPOCHS, ADDRESS
from helpers.constants import SERVER_PORT, MESSAGE_START_TRAINING, MESSAGE_FL_UPDATE_ENCRYPTED, CHUNK_SIZE


class FedAvgNode:

    def __init__(self, address, port, client_type, dataset, x_train, y_train, x_test, y_test):
        self.app = FastAPI()
        self.port = port
        self.address = address

        self.dataset = dataset
        self.client_type = client_type

        self.model = None
        self.epochs = EPOCHS

        self.record = list()

        self.round = 0
        self.current_accuracy = 0
        self.current_training_time = 0

        self.server_public_key = get_public_key('server')

        self.X_train, self.y_train, self.X_test, self.y_test = x_train, y_train, x_test, y_test

        @self.app.post("/message")
        def message(data: dict):
            print(f"PORT {self.port} RECEIVED: {data['message']}")
            if data["message"] == MESSAGE_START_TRAINING:
                self.start_training(data)

            elif data["message"] == MESSAGE_END_SESSION:
                self.end_session(data)

            return {"status": "ok"}

    def start(self):
        if check_port(self.address, self.port):
            terminate_process_on_port(self.port)

        uvicorn.run(self.app, host="0.0.0.0", port=self.port)

    @staticmethod
    def send_to_node(address, port, data):
        post_with_retries(
            data=data,
            url=f"http://{address}:{port}/message",
            max_retries=3
        )

    def start_training(self, global_model):
        self.round += 1
        self.model = tf.keras.models.model_from_json(global_model["model_architecture"])
        self.model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        self.model.set_weights(decode_layer(global_model["model_weights"]))

        cb = TimingCallback()

        self.model.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=10, callbacks=[cb], verbose=False)
        _, self.current_accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        self.current_training_time = sum(cb.logs)

        self.send_updates()

    def send_updates(self):
        model_weights = dict()
        for layer in self.model.layers:
            if layer.trainable_weights:
                model_weights[layer.name] = layer.weights

        json_str = json.dumps(model_weights, cls=NumpyEncoder)

        value_bytes = json_str.encode('utf-8')
        num_chunks = (len(value_bytes) + CHUNK_SIZE - 1) // CHUNK_SIZE

        weight_chunks = []
        for i in range(num_chunks):
            start = i * CHUNK_SIZE
            end = start + CHUNK_SIZE
            chunk = value_bytes[start:end]
            weight_chunks.append(chunk)

        encrypted_messages = []
        for json_byte_chunk in weight_chunks:
            encrypted_messages.append(
                self.server_public_key.encrypt(
                    json_byte_chunk,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
            )
        self.record.append({
            'round': self.round,
            'accuracy': self.current_accuracy,
            'training': self.current_training_time,
        })

        data = {
            "port": self.port,
            "message": MESSAGE_FL_UPDATE_ENCRYPTED,
            "model_weights": encode_layer(encrypted_messages),
        }

        self.send_to_node(ADDRESS, SERVER_PORT, data)

    def end_session(self, data):
        model_weights = decode_layer(data['model_weights'])
        self.model.set_weights(model_weights)
        self.disconnect()

    def disconnect(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        output_folder = current_dir + f"/resources/results/{self.client_type}/{self.dataset}"
        os.makedirs(output_folder, exist_ok=True)
        csv_filename = f'client_{self.port - CLIENT_PORT}.csv'
        csv_path = os.path.join(output_folder, csv_filename)
        pd.DataFrame(self.record).to_csv(csv_path, index=False, header=True)


if __name__ == "__main__":

    DATASET = str(sys.argv[1])
    print(f"DATASET: {DATASET}")

    indexes = fetch_index(DATASET)
    (x_train, y_train), (x_test, y_test) = fetch_dataset(DATASET)

    nodes = []
    ports = []
    threads = []

    server = Server(
        server_id=SERVER_ID,
        address=ADDRESS,
        port=SERVER_PORT,
        max_nodes=NODES,
        client_type='fed_avg_encrypted',
        dataset=DATASET,
        indexes=indexes,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test
    )
    server_thread = threading.Thread(target=server.start)

    for i in range(1, NODES + 1):
        X_train, Y_train, X_test, Y_test = get_dataset(
            indexes[i - 1],
            DATASET,
            x_train,
            y_train,
            x_test,
            y_test
        )

        node = FedAvgNode(
            address=ADDRESS,
            port=CLIENT_PORT + i,
            client_type="fed_avg_encrypted",
            dataset=DATASET,
            x_train=X_train,
            y_train=Y_train,
            x_test=X_test,
            y_test=Y_test
        )
        ports.append(CLIENT_PORT + i)
        nodes.append(node)

    server_thread.start()
    for node in nodes:
        time.sleep(2)
        t = threading.Thread(target=node.start)
        t.start()
        threads.append(t)

    server.start_round(ports)

    server_thread.join()
    for t in threads:
        t.join()
