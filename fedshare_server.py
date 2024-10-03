import os
import time
import numpy as np
import uvicorn
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI
from timeit import default_timer as timer

from helpers.utils import check_port, terminate_process_on_port, get_dataset, combine_csv_files
from helpers.utils import post_with_retries, encode_layer, decode_layer, get_lenet5_classification, get_private_key

from helpers.constants import MESSAGE_FEDSHARE_SHARE, ADDRESS, MESSAGE_MODEL_SHARE, SERVER_PORT, ROUNDS


class FedShareServer:
    def __init__(self, address, port, max_nodes, client_type, dataset, indexes, x_train, y_train, x_test,
                 y_test):
        self.app = FastAPI()
        self.port = port
        self.address = address
        self.connected_nodes = 0
        self.max_nodes = max_nodes
        self.scotch_servers = list()

        self.start_time = None
        self.end_time = None
        self.pending_nodes = set()
        self.average_weights = dict()
        _, _, self.X_test, self.y_test = get_dataset(
            indexes[self.port - SERVER_PORT],
            dataset,
            x_train,
            y_train,
            x_test,
            y_test
        )

        self.global_model = get_lenet5_classification(dataset)
        self.max_rounds = ROUNDS
        self.round = 0
        self.training_completed_count = 0
        self.client_type = client_type
        self.dataset = dataset

        self.record = list()
        self.current_accuracy = 0
        self.threshold = 0
        self.share_count = 0

        self.shares = dict()
        self.nodes = []

        self.private_key = get_private_key('server')

        @self.app.post("/message")
        def message(data: dict):
            print(f"SERVER RECEIVED: {data['message']} from PORT: {data['port']}")

            if data["message"] == MESSAGE_MODEL_SHARE:
                self.accept_shares(data["model_share"], data["data_size"])

            return {"status": "ok"}

    def start(self):
        if check_port(self.address, self.port):
            terminate_process_on_port(self.port)

        uvicorn.run(self.app, host="0.0.0.0", port=self.port)

    def send_to_node(self, data, port=None):
        if port is None:
            for port in self.nodes:
                post_with_retries(
                    data=data,
                    url=f"http://{ADDRESS}:{port}/message",
                    max_retries=3
                )
                time.sleep(2)
        else:
            post_with_retries(
                data=data,
                url=f"http://{ADDRESS}:{port}/message",
                max_retries=3
            )

    def accept_shares(self, data, size):

        self.start_time = timer()

        for layer in data.keys():
            weight_bias = decode_layer(data[layer])
            total_dataset_size = sum([size / self.max_nodes] * self.max_nodes)

            self.shares[layer] = [[], []]
            self.shares[layer][0].append(weight_bias[0] * (size / total_dataset_size))
            self.shares[layer][1].append(weight_bias[1] * (size / total_dataset_size))

        self.share_count += 1

        if self.share_count == self.max_nodes:
            self.share_count = 0
            self.reassemble_shares()

    def reassemble_shares(self):
        self.start_time = timer()

        for layer in self.shares.keys():
            self.average_weights[layer] = [[], []]
            self.average_weights[layer][0] = np.sum((self.shares[layer][0]), axis=0)
            self.average_weights[layer][1] = np.sum((self.shares[layer][1]), axis=0)
        self.evaluate()

    def evaluate(self):

        model_weights = dict()
        for layer in self.global_model.layers:
            if layer.trainable_weights:
                layer.set_weights(self.average_weights[layer.name])
                model_weights[layer.name] = encode_layer(self.average_weights[layer.name])
                self.shares[layer.name] = [[], []]
                self.average_weights[layer] = [None, None]

        self.global_model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
                                  loss='categorical_crossentropy', metrics=['accuracy'])
        _, self.current_accuracy = self.global_model.evaluate(self.X_test, self.y_test, verbose=0)
        self.end_time = timer() - self.start_time
        print(f"FEDSHARE SERVER : {self.port - SERVER_PORT}")
        print('Accuracy: ', self.current_accuracy)
        print(f'Time: {self.end_time}')

        self.record.append({
            'round': self.round + 1,
            'accuracy': self.current_accuracy,
            'fl': self.end_time,
        })

        data = {
            "port": self.port,
            "message": MESSAGE_FEDSHARE_SHARE,
            "model_share": model_weights,
        }
        self.send_to_node(data, SERVER_PORT)

    def disconnect(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        output_folder = current_dir + f"/resources/results/{self.client_type}/{self.dataset}"
        os.makedirs(output_folder, exist_ok=True)
        csv_filename = f'server_{self.port - SERVER_PORT}.csv'
        csv_path = os.path.join(output_folder, csv_filename)
        pd.DataFrame(self.record).to_csv(csv_path, index=False, header=True)

        combine_csv_files(f"{self.client_type}", f"{self.dataset}")
        # terminate_process_on_port(self.port)
