import os
import time
import uvicorn
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI
from timeit import default_timer as timer

from helpers.utils import check_port, terminate_process_on_port, combine_csv_files
from helpers.utils import post_with_retries, encode_layer, decode_layer, get_lenet5_classification, get_private_key

from helpers.constants import MESSAGE_SCOTCH_SHARE, ADDRESS, MESSAGE_MODEL_SHARE
from helpers.constants import MESSAGE_START_ASSEMBLY, SERVER_PORT, ROUNDS, MESSAGE_ASSEMBLY_COMPLETED


class ScotchServer:

    def __init__(self, address, port, max_nodes, client_type, dataset, x_test, y_test):
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
        self.X_test, self.y_test = x_test, y_test
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
            print(f"SERVER RECEIVED ({self.port}): {data['message']} from PORT: {data['port']}")

            if data["message"] == MESSAGE_MODEL_SHARE:
                self.accept_shares(data["model_share"])
            elif data["message"] == MESSAGE_ASSEMBLY_COMPLETED:
                self.end_round()

            return {"status": "ok"}

    def start(self, nodes=None):
        if nodes:
            self.nodes = nodes
            for layer in self.global_model.layers:
                if layer.trainable_weights:
                    self.average_weights[layer.name] = [[], []]
                    self.shares[layer.name] = [[], []]

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

    def accept_shares(self, data):

        self.start_time = timer()
        for layer in data.keys():
            weight_bias = decode_layer(data[layer])
            self.shares[layer][0].append(weight_bias[0])
            self.shares[layer][1].append(weight_bias[1])

        self.share_count += 1

        if self.share_count == len(self.nodes):
            # print(f"RECEIVED SHARES (SERVER {self.port}): {self.share_count}")
            self.share_count = 0
            self.reassemble_shares()

    def reassemble_shares(self):
        self.start_time = timer()

        for layer in self.shares.keys():
            temp_weight_bias = [None, None]
            temp_weight_bias[0] = np.sum((self.shares[layer][0]), axis=0)
            temp_weight_bias[1] = np.sum((self.shares[layer][1]), axis=0)

            if len(self.average_weights[layer][0]) == 0 and len(self.average_weights[layer][1]) == 0:
                self.average_weights[layer][0] = temp_weight_bias[0] / len(self.nodes)
                self.average_weights[layer][1] = temp_weight_bias[1] / len(self.nodes)
            else:
                self.average_weights[layer][0] += temp_weight_bias[0] / len(self.nodes)
                self.average_weights[layer][1] += temp_weight_bias[1] / len(self.nodes)

        model_weights = dict()
        for layer in self.global_model.layers:
            if layer.trainable_weights:
                layer.set_weights(self.average_weights[layer.name])
                model_weights[layer.name] = encode_layer(self.average_weights[layer.name])
                self.shares[layer.name] = [[], []]
                self.average_weights[layer] = [None, None]

        self.evaluate()

        # for node in self.nodes:
        #     data = {
        #         "port": self.port,
        #         "message": MESSAGE_SCOTCH_SHARE,
        #         "model_share": model_weights,
        #     }
        #     self.send_to_node(data, node)
        #
        # self.end_round()

    def evaluate(self):
        # print(f"RECEIVED SHARES (SERVER {self.port}): {len(self.shares['conv2d_0'][0])}")
        self.global_model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
                                  loss='categorical_crossentropy', metrics=['accuracy'])
        _, self.current_accuracy = self.global_model.evaluate(self.X_test, self.y_test, verbose=0)
        self.end_time = timer() - self.start_time
        print(f"(Round {self.round + 1}) SERVER : {self.port - SERVER_PORT}")
        print('Accuracy: ', self.current_accuracy)
        print(f'Time: {self.end_time}')

        self.record.append({
            'round': self.round + 1,
            'accuracy': self.current_accuracy,
            'fl': self.end_time,
        })

    def end_round(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        output_folder = current_dir + f"/resources/results/{self.client_type}/{self.dataset}"
        os.makedirs(output_folder, exist_ok=True)
        csv_filename = f'server_{self.port - SERVER_PORT}.csv'
        csv_path = os.path.join(output_folder, csv_filename)
        pd.DataFrame(self.record).to_csv(csv_path, index=False, header=True)
        combine_csv_files(f"{self.client_type}", f"{self.dataset}")

        print("ROUND ENDED")

        self.round += 1
        if self.round <= self.max_rounds:
            for node in self.nodes:
                data = {
                    "port": self.port,
                    "message": MESSAGE_START_ASSEMBLY,
                }
                self.send_to_node(data, node)
