import os
import time
import numpy as np
import uvicorn
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI
from timeit import default_timer as timer

from helpers.utils import check_port, terminate_process_on_port, get_dataset, combine_find_mean
from helpers.utils import post_with_retries, encode_layer, decode_layer, get_lenet5_classification, get_private_key

from helpers.constants import MESSAGE_START_TRAINING, ADDRESS, MESSAGE_FEDSHARE_SHARE, ROUNDS, MESSAGE_END_SESSION


class FedShareLeadServer:
    def __init__(self, address, port, client_type, dataset, indexes, x_train, y_train, x_test, y_test):
        self.app = FastAPI()
        self.port = port
        self.address = address
        self.connected_nodes = 0
        self.scotch_servers = list()

        self.start_time = None
        self.end_time = None
        self.pending_nodes = set()
        self.average_weights = dict()
        _, _, self.X_test, self.y_test = get_dataset(
            indexes[0],
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
        self.servers = []
        self.nodes = []

        self.private_key = get_private_key('server')

        @self.app.post("/message")
        def message(data: dict):
            print(f"SERVER RECEIVED: {data['message']} from PORT: {data['port']}")

            if data["message"] == MESSAGE_FEDSHARE_SHARE:
                self.accept_shares(data["model_share"])

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

    def start_round(self, servers=None, nodes=None):
        print(f'Starting round ({self.round + 1})')

        if servers and nodes:
            self.nodes = nodes
            self.servers = servers

        self.start_time = timer()
        self.pending_nodes = self.nodes.copy()

        for layer in self.global_model.layers:
            if layer.trainable_weights:
                self.average_weights[layer.name] = [[], []]
                self.shares[layer.name] = [[], []]

        data = {
            "port": "SERVER",
            "servers": self.servers,
            "message": MESSAGE_START_TRAINING,
            "model_architecture": self.global_model.to_json(),
            "model_weights": encode_layer(self.global_model.get_weights()),
        }
        self.send_to_node(data)

    def accept_shares(self, data):
        for layer in data.keys():
            weight_bias = decode_layer(data[layer])
            self.shares[layer][0].append(weight_bias[0])
            self.shares[layer][1].append(weight_bias[1])

        self.share_count += 1

        if self.share_count == len(self.servers):
            self.share_count = 0
            self.apply_updates()

    def apply_updates(self):

        for layer in self.shares.keys():
            self.average_weights[layer][0] = np.sum((self.shares[layer][0]), axis=0)
            self.average_weights[layer][1] = np.sum((self.shares[layer][1]), axis=0)

        for layer in self.global_model.layers:
            if layer.trainable_weights:
                layer.set_weights(self.average_weights[layer.name])

        self.evaluate()

    def evaluate(self):
        self.global_model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
                                  loss='categorical_crossentropy', metrics=['accuracy'])
        _, self.current_accuracy = self.global_model.evaluate(self.X_test, self.y_test, verbose=0)
        self.end_time = timer() - self.start_time
        print(f"LEAD SERVER")
        print('Accuracy: ', self.current_accuracy)
        print(f'Time: {self.end_time}')

        self.record.append({
            'round': self.round + 1,
            'accuracy': self.current_accuracy,
            'fl': self.end_time,
        })

        self.end_round()

    def end_round(self):
        print("ROUND ENDED")
        self.round += 1
        if self.round < self.max_rounds:
            self.start_round()
        else:
            self.end_session()

    def end_session(self):
        print("SESSION ENDED")
        data = {
            "port": "SERVER",
            "message": MESSAGE_END_SESSION,
            "model_weights": encode_layer(self.global_model.get_weights()),
        }
        self.send_to_node(data)

        current_dir = os.path.dirname(os.path.realpath(__file__))
        output_folder = current_dir + f"/resources/results/{self.client_type}/{self.dataset}"
        os.makedirs(output_folder, exist_ok=True)
        csv_filename = f'lead_server.csv'
        csv_path = os.path.join(output_folder, csv_filename)
        pd.DataFrame(self.record).to_csv(csv_path, index=False, header=True)
        combine_find_mean(f"{self.client_type}", f"{self.dataset}")
        terminate_process_on_port(self.port)
