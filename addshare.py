import os
import sys
import time
import uvicorn
import threading
import numpy as np
import pandas as pd
import tensorflow as tf
from server import Server
from fastapi import FastAPI
from timeit import default_timer as timer

from helpers.utils import check_port, terminate_process_on_port, decode_layer, TimingCallback, encode_layer
from helpers.utils import fetch_dataset, fetch_index, get_dataset, post_with_retries, generate_additive_shares

from helpers.constants import MESSAGE_FL_UPDATE, CLIENT_PORT, MESSAGE_MODEL_SHARE, MESSAGE_SHARING_COMPLETE
from helpers.constants import EPOCHS, ADDRESS, SERVER_PORT, MESSAGE_START_TRAINING, MESSAGE_END_SESSION, SERVER_ID
from helpers.constants import MESSAGE_START_ASSEMBLY, NODES, MESSAGE_START_SECRET_SHARING, MESSAGE_TRAINING_COMPLETED


class AddShareNode:

    def __init__(self, address, port, client_type, dataset, x_train, y_train, x_test, y_test):
        self.app = FastAPI()
        self.port = port
        self.address = address

        self.dataset = dataset
        self.client_type = client_type

        self.model = None
        self.epochs = EPOCHS

        self.own_shares = dict()
        self.other_shares = dict()

        self.fl_nodes = list()
        self.share_count = 0

        self.start_time = None
        self.secret_sharing_time = 0.0

        self.record = list()

        self.round = 0
        self.current_accuracy = 0
        self.current_training_time = 0

        self.X_train, self.y_train, self.X_test, self.y_test = x_train, y_train, x_test, y_test

        @self.app.post("/message")
        def message(data: dict):
            print(f"PORT {self.port} RECEIVED: {data['message']} from {data['port']}")
            if data["message"] == MESSAGE_START_TRAINING:
                self.start_training(data)

            elif data["message"] == MESSAGE_START_SECRET_SHARING:
                self.start_secret_sharing()

            elif data["message"] == MESSAGE_END_SESSION:
                self.end_session(data)

            elif data["message"] == MESSAGE_MODEL_SHARE:
                self.accept_shares(data['model_share'])

            elif data["message"] == MESSAGE_START_ASSEMBLY:
                self.reassemble_shares()

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

    def start_training(self, data):

        self.fl_nodes = [port for port in data["nodes"] if port != self.port]
        self.round += 1
        self.model = tf.keras.models.model_from_json(data["model_architecture"])
        self.model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        self.model.set_weights(decode_layer(data["model_weights"]))

        cb = TimingCallback()

        self.model.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=10, callbacks=[cb], verbose=False)
        _, self.current_accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        self.current_training_time = sum(cb.logs)

        for layer in self.model.layers:
            if layer.trainable_weights:
                self.own_shares[layer.name] = [[], []]
                self.other_shares[layer.name] = [None, None]

        data = {
            "port": self.port,
            "message": MESSAGE_TRAINING_COMPLETED
        }

        self.send_to_node(address=ADDRESS, port=SERVER_PORT, data=data)

    def start_secret_sharing(self):
        self.start_time = timer()
        shares = int(len(self.fl_nodes) + 1)

        for layer in self.model.layers:
            if layer.trainable_weights:
                weight_shares = list(generate_additive_shares(layer.weights[0], shares))
                bias_shares = list(generate_additive_shares(layer.weights[1], shares))

                self.own_shares[layer.name] = [[], []]
                self.own_shares[layer.name][0].append(weight_shares.pop())
                self.own_shares[layer.name][1].append(bias_shares.pop())

                self.other_shares[layer.name] = [None, None]
                self.other_shares[layer.name][0] = weight_shares
                self.other_shares[layer.name][1] = bias_shares

        self.share_count += 1

        self.secret_sharing_time = timer() - self.start_time
        self.start_exchanging_shares()

    def start_exchanging_shares(self):
        self.start_time = timer()
        for client in self.fl_nodes:
            layer_weights = dict()

            for layer in self.other_shares.keys():
                weight_bias = [None, None]
                weight_bias[0] = self.other_shares[layer][0].pop()
                weight_bias[1] = self.other_shares[layer][1].pop()

                layer_weights[layer] = encode_layer(weight_bias)

            data = {
                "port": self.port,
                "message": MESSAGE_MODEL_SHARE,
                "model_share": layer_weights,
            }

            self.send_to_node(data=data, address=ADDRESS, port=client)

        self.secret_sharing_time = self.secret_sharing_time + (timer() - self.start_time)

        if self.share_count == int(len(self.fl_nodes) + 1):
            self.share_count = 0
            data = {
                "port": self.port,
                "message": MESSAGE_SHARING_COMPLETE,
            }
            self.send_to_node(data=data, address=ADDRESS, port=SERVER_PORT)

    def accept_shares(self, data):

        self.start_time = timer()
        for layer in data.keys():
            weight_bias = decode_layer(data[layer])
            self.own_shares[layer][0].append(weight_bias[0])
            self.own_shares[layer][1].append(weight_bias[1])

        self.share_count += 1

        self.secret_sharing_time = self.secret_sharing_time + (timer() - self.start_time)

        if self.share_count == int(len(self.fl_nodes) + 1):
            self.share_count = 0
            data = {
                "port": self.port,
                "message": MESSAGE_SHARING_COMPLETE,
            }
            self.send_to_node(data=data, address=ADDRESS, port=SERVER_PORT)

    def reassemble_shares(self):
        self.start_time = timer()
        layer_weights = dict()

        for layer in self.own_shares.keys():
            temp_weight_bias = [None, None]
            temp_weight_bias[0] = np.sum((self.own_shares[layer][0]), axis=0)
            temp_weight_bias[1] = np.sum((self.own_shares[layer][1]), axis=0)
            layer_weights[layer] = encode_layer(temp_weight_bias)

        self.secret_sharing_time = self.secret_sharing_time + (timer() - self.start_time)

        self.record.append({
            'round': self.round,
            'accuracy': self.current_accuracy,
            'training': self.current_training_time,
            'secret_sharing': self.secret_sharing_time
        })

        current_dir = os.path.dirname(os.path.realpath(__file__))
        output_folder = current_dir + f"/resources/results/{self.client_type}/{self.dataset}"
        os.makedirs(output_folder, exist_ok=True)
        csv_filename = f'client_{self.port - CLIENT_PORT}.csv'
        csv_path = os.path.join(output_folder, csv_filename)
        pd.DataFrame(self.record).to_csv(csv_path, index=False, header=True)


        data = {
            "port": self.port,
            "message": MESSAGE_FL_UPDATE,
            "model_weights": layer_weights,
        }
        self.send_to_node(address=ADDRESS, port=SERVER_PORT, data=data)

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
        client_type='addshare',
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

        node = AddShareNode(
            address=ADDRESS,
            port=CLIENT_PORT + i,
            client_type="addshare",
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
