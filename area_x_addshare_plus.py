import os
import time
import json
import uvicorn
import threading
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI
from timeit import default_timer as timer

from helpers import constants
from area_x_server import AreaXAddsharePlusServer
from helpers.utils import generate_additive_shares, post_with_retries, get_area_x_dataset
from helpers.utils import check_port, terminate_process_on_port, TimingCallback, encode_layer, decode_layer


class AreaXAddSharePlusNode:

    def __init__(self, address, port, client_type, pruning_type, dataset, x_train, y_train, x_test, y_test):
        self.app = FastAPI()
        self.port = port
        self.address = address

        self.dataset = dataset
        self.client_type = client_type
        self.pruning_type = pruning_type

        self.model = None
        self.epochs = constants.EPOCHS

        self.indexes = list()
        self.own_shares = dict()
        self.other_shares = dict()

        self.fl_nodes = list()
        self.share_count = 0

        self.start_time = None
        self.secret_sharing_time = 0.0
        self.current_training_time = 0

        self.record = list()
        self.round, self.mae, self.rmse, self.mape = 0, 0, 0, 0
        self.X_train, self.y_train, self.X_test, self.y_test = x_train, y_train, x_test, y_test

        @self.app.post("/message")
        def message(data: dict):
            print(f"PORT {self.port} RECEIVED: {data['message']} from {data['port']}")
            if data["message"] == constants.MESSAGE_START_TRAINING:
                self.start_training(data)

            elif data["message"] == constants.MESSAGE_START_SECRET_SHARING:
                self.start_secret_sharing()

            elif data["message"] == constants.MESSAGE_END_SESSION:
                self.end_session(data)

            elif data["message"] == constants.MESSAGE_MODEL_SHARE:
                self.accept_shares(data['model_share'])

            elif data["message"] == constants.MESSAGE_START_ASSEMBLY:
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
        self.indexes = json.loads(data["indexes"])
        self.round += 1
        self.model = tf.keras.models.model_from_json(data["model_architecture"])
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            loss=tf.keras.losses.mae,
            metrics=[tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanAbsolutePercentageError()]
        )
        self.model.set_weights(decode_layer(data["model_weights"]))

        cb = TimingCallback()
        self.model.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=10, callbacks=[cb], verbose=False)
        self.mae, self.rmse, self.mape = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        self.current_training_time = sum(cb.logs)

        for layer in self.model.layers:
            if layer.trainable_weights:
                self.own_shares[layer.name] = [[], []]
                self.other_shares[layer.name] = [None, None]

        data = {
            "port": self.port,
            "message": constants.MESSAGE_TRAINING_COMPLETED
        }

        self.send_to_node(address=constants.ADDRESS, port=constants.SERVER_PORT, data=data)

    def start_secret_sharing(self):
        self.start_time = timer()
        shares = int(len(self.fl_nodes) + 1)

        for layer in self.model.layers:
            if layer.trainable_weights:
                # get random indexes
                selected_kernel_index = tuple(np.array(li) for li in self.indexes[layer.name][0])
                selected_bias_index = tuple(np.array(li) for li in self.indexes[layer.name][1])

                # get selected columns
                selected_kernels = layer.get_weights()[0][selected_kernel_index]
                selected_bias = layer.get_weights()[1][selected_bias_index]

                # generate additive shares of selected weights
                weight_shares = list(generate_additive_shares(selected_kernels, NODES))
                bias_shares = list(generate_additive_shares(selected_bias, NODES))

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
                "message": constants.MESSAGE_MODEL_SHARE,
                "model_share": layer_weights,
            }

            self.send_to_node(data=data, address=constants.ADDRESS, port=client)

        self.secret_sharing_time = self.secret_sharing_time + (timer() - self.start_time)

        if self.share_count == int(len(self.fl_nodes) + 1):
            self.share_count = 0
            data = {
                "port": self.port,
                "message": constants.MESSAGE_SHARING_COMPLETE,
            }
            self.send_to_node(data=data, address=constants.ADDRESS, port=constants.SERVER_PORT)

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
                "message": constants.MESSAGE_SHARING_COMPLETE,
            }
            self.send_to_node(data=data, address=constants.ADDRESS, port=constants.SERVER_PORT)

    def reassemble_shares(self):
        self.start_time = timer()
        layer_weights = dict()

        for layer in self.own_shares.keys():
            kernel = np.sum((self.own_shares[layer][0]), axis=0)
            bias = np.sum((self.own_shares[layer][1]), axis=0)

            selected_kernel_index = tuple(np.array(li) for li in self.indexes[layer][0])
            selected_bias_index = tuple(np.array(li) for li in self.indexes[layer][1])

            # Get original model weights
            temp_weight_bias = [
                self.model.get_layer(layer).get_weights()[0],
                self.model.get_layer(layer).get_weights()[1]
            ]

            # Replace original selected weights with assembled additive shares
            temp_weight_bias[0][selected_kernel_index] = kernel
            temp_weight_bias[1][selected_bias_index] = bias
            layer_weights[layer] = encode_layer(temp_weight_bias)

        self.secret_sharing_time = self.secret_sharing_time + (timer() - self.start_time)

        self.record.append({
            'round': self.round,
            'loss': self.mae,
            'rmse': self.rmse,
            'mape': self.mape,
            'training': self.current_training_time,
            'secret_sharing': self.secret_sharing_time
        })

        data = {
            "port": self.port,
            "message": constants.MESSAGE_FL_UPDATE,
            "model_weights": layer_weights,
        }
        self.send_to_node(address=constants.ADDRESS, port=constants.SERVER_PORT, data=data)

    def end_session(self, data):
        model_weights = decode_layer(data['model_weights'])
        self.model.set_weights(model_weights)
        self.disconnect()

    def disconnect(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        output_folder = current_dir + f"/resources/results/{self.client_type}_{self.pruning_type}/{self.dataset}"
        os.makedirs(output_folder, exist_ok=True)
        csv_filename = f'client_{self.port - constants.CLIENT_PORT}.csv'
        csv_path = os.path.join(output_folder, csv_filename)
        pd.DataFrame(self.record).to_csv(csv_path, index=False, header=True)


if __name__ == "__main__":
    NODES = len(constants.FIELDS)
    DATASET = "area_x"  # str(sys.argv[1])
    SELECTION_TYPE = constants.MAGNITUDE  # str(sys.argv[2])
    print(f"DATASET: {DATASET}, SELECTION TYPE: {SELECTION_TYPE}")

    x, y, _, _ = get_area_x_dataset(9)

    nodes = []
    ports = []
    threads = []

    server = AreaXAddsharePlusServer(
        server_id=constants.SERVER_ID,
        address=constants.ADDRESS,
        port=constants.SERVER_PORT,
        max_nodes=NODES,
        client_type='addshare_plus',
        pruning_type=SELECTION_TYPE,
        dataset=DATASET,
        x=x,
        y=y
    )
    server_thread = threading.Thread(target=server.start)

    for i in constants.FIELDS:
        X_train, Y_train, X_test, Y_test = get_area_x_dataset(i)

        node = AreaXAddSharePlusNode(
            address=constants.ADDRESS,
            port=constants.CLIENT_PORT + int(i),
            client_type="addshare_plus",
            pruning_type=SELECTION_TYPE,
            dataset=DATASET,
            x_train=X_train,
            y_train=Y_train,
            x_test=X_test,
            y_test=Y_test
        )
        ports.append(constants.CLIENT_PORT + int(i))
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
