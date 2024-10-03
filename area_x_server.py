import os
import time
import json
import uvicorn
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI
from timeit import default_timer as timer

from helpers import constants
from helpers.utils import random_weight_selection, magnitude_weight_selection, obd_weight_selection
from helpers.utils import check_port, terminate_process_on_port, decode_layer, combine_find_mean_regression
from helpers.utils import get_regression_model, regularization_weight_selection, post_with_retries, encode_layer


class AreaXAddsharePlusServer:
    def __init__(self, server_id, address, port, max_nodes, client_type, pruning_type, dataset, x, y):
        self.id = server_id
        self.app = FastAPI()
        self.port = port
        self.address = address
        self.connected_nodes = 0
        self.max_nodes = max_nodes
        self.nodes = list()

        self.start_time = None
        self.end_time = None
        self.pending_nodes = set()
        self.average_weights = dict()
        self.X, self.y = x, y

        self.global_model = get_regression_model()
        self.max_rounds = constants.ROUNDS
        self.round = 0
        self.training_completed_count = 0
        self.client_type = client_type
        self.pruning_type = pruning_type
        self.dataset = dataset

        self.record = list()
        self.loss, self.rmse, self.mape = 0, 0, 0
        self.threshold = 0

        @self.app.post("/message")
        def message(data: dict):
            print(f"SERVER RECEIVED: {data['message']} from PORT: {data['port']}")

            if data["message"] == constants.MESSAGE_FL_UPDATE:
                self.fl_update(data["port"], data["model_weights"])

            elif data["message"] == constants.MESSAGE_TRAINING_COMPLETED:
                self.start_secret_sharing()

            elif data["message"] == constants.MESSAGE_SHARING_COMPLETE:
                self.start_assembly(data["port"])

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
                    url=f"http://{constants.ADDRESS}:{port}/message",
                    max_retries=3
                )
                time.sleep(2)
        else:
            post_with_retries(
                data=data,
                url=f"http://{constants.ADDRESS}:{port}/message",
                max_retries=3
            )

    def start_round(self, nodes=None):
        if nodes:
            self.nodes = nodes

        print(f'Starting round ({self.round + 1})')

        indexes = {}
        self.start_time = timer()
        self.pending_nodes = self.nodes.copy()
        for layer in self.global_model.layers:
            if layer.trainable_weights:
                if self.pruning_type == constants.RANDOM:
                    kernel_indices = random_weight_selection(layer.get_weights()[0], constants.THRESHOLD)
                    bias_indices = random_weight_selection(layer.get_weights()[1], constants.THRESHOLD)
                    indexes[layer.name] = [kernel_indices, bias_indices]
                elif self.pruning_type == constants.MAGNITUDE:
                    kernel_indices = magnitude_weight_selection(layer.get_weights()[0], constants.THRESHOLD)
                    bias_indices = magnitude_weight_selection(layer.get_weights()[1], constants.THRESHOLD)
                    indexes[layer.name] = [kernel_indices, bias_indices]
                elif self.pruning_type == constants.OBD:
                    kernel_indices = obd_weight_selection(
                        self.global_model,
                        self.X,
                        self.y,
                        layer.trainable_weights[0],
                        constants.THRESHOLD
                    )
                    bias_indices = obd_weight_selection(
                        self.global_model,
                        self.X,
                        self.y,
                        layer.trainable_weights[1],
                        constants.THRESHOLD
                    )
                    indexes[layer.name] = [kernel_indices, bias_indices]
                elif self.pruning_type == constants.L2R:
                    kernel_indices = regularization_weight_selection(
                        self.global_model,
                        self.X,
                        self.y,
                        self.pruning_type,
                        layer.trainable_weights[0],
                        constants.THRESHOLD
                    )
                    bias_indices = regularization_weight_selection(
                        self.global_model,
                        self.X,
                        self.y,
                        self.pruning_type,
                        layer.trainable_weights[1],
                        constants.THRESHOLD
                    )
                    indexes[layer.name] = [kernel_indices, bias_indices]

                self.average_weights[layer.name] = [[], []]

        data = {
            "port": "SERVER",
            "nodes": self.nodes,
            "indexes": json.dumps(indexes),
            "message": constants.MESSAGE_START_TRAINING,
            "model_architecture": self.global_model.to_json(),
            "model_weights": encode_layer(self.global_model.get_weights()),
        }
        self.send_to_node(data)

    def fl_update(self, node, data):

        for layer in data.keys():
            temp_weight = decode_layer(data[layer])

            if len(self.average_weights[layer][0]) == 0 and len(self.average_weights[layer][1]) == 0:
                self.average_weights[layer][0] = temp_weight[0] / len(self.nodes)
                self.average_weights[layer][1] = temp_weight[1] / len(self.nodes)
            else:
                self.average_weights[layer][0] += temp_weight[0] / len(self.nodes)
                self.average_weights[layer][1] += temp_weight[1] / len(self.nodes)

        self.pending_nodes.remove(node)
        if not self.pending_nodes:
            self.apply_updates()

    def apply_updates(self):
        for layer in self.global_model.layers:
            if layer.trainable_weights:
                layer.set_weights(self.average_weights[layer.name])
        self.evaluate()

    def evaluate(self):
        self.global_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            loss=tf.keras.losses.mae,
            metrics=[tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanAbsolutePercentageError()]
        )
        self.loss, self.rmse, self.mape = self.global_model.evaluate(self.X, self.y, verbose=0)
        self.end_time = timer() - self.start_time
        print('Loss: ', self.loss)
        print('Mean Squared Error: ', self.rmse)
        print('Mean Absolute Percentage Error: ', self.mape)
        print(f'Round ({self.round + 1}) Time: {self.end_time}')

        self.record.append({
            'round': self.round + 1,
            'loss': self.loss,
            'rmse': self.rmse,
            'mape': self.mape,
            'fl_time': self.end_time,
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
            "message": constants.MESSAGE_END_SESSION,
            "model_weights": encode_layer(self.global_model.get_weights()),
        }

        current_dir = os.path.dirname(os.path.realpath(__file__))
        output_folder = current_dir + f"/resources/results/{self.client_type}_{self.pruning_type}/{self.dataset}"
        os.makedirs(output_folder, exist_ok=True)
        csv_filename = 'server.csv'
        csv_path = os.path.join(output_folder, csv_filename)
        pd.DataFrame(self.record).to_csv(csv_path, index=False, header=True)

        self.send_to_node(data)
        combine_find_mean_regression(f"{self.client_type}_{self.pruning_type}", f"{self.dataset}")
        terminate_process_on_port(self.port)

    def start_assembly(self, port):
        data = {
            "port": "SERVER",
            "message": constants.MESSAGE_START_ASSEMBLY,
        }
        self.send_to_node(data, port=port)

    def start_secret_sharing(self):
        self.training_completed_count += 1
        if self.training_completed_count == len(self.nodes):
            self.training_completed_count = 0
            data = {
                "port": "SERVER",
                "message": constants.MESSAGE_START_SECRET_SHARING,
            }
            self.send_to_node(data)
