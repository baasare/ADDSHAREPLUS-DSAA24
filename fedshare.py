import os
import uvicorn
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI
from timeit import default_timer as timer

from helpers.utils import post_with_retries, generate_additive_shares, TimingCallback, f_to_i, i_to_f
from helpers.utils import check_port, terminate_process_on_port, decode_layer, encode_layer, get_lenet5_classification

from helpers.constants import MESSAGE_MODEL_SHARE, MESSAGE_END_SESSION
from helpers.constants import EPOCHS, ADDRESS, CLIENT_PORT, MESSAGE_START_TRAINING


class FedShareNode:

    def __init__(self, address, port, client_type, dataset, x_train, y_train, x_test, y_test):
        self.app = FastAPI()
        self.port = port
        self.address = address

        self.dataset = dataset
        self.client_type = client_type
        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

        self.model = get_lenet5_classification(dataset)
        self.epochs = EPOCHS

        self.fedshare_servers = list()
        self.scotch_servers_shares = dict()
        self.model_shares = dict()
        self.share_count = 0

        self.secret_sharing_time = 0.0
        self.f_to_i_v = np.vectorize(f_to_i)
        self.i_to_f_v = np.vectorize(i_to_f)

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

            elif data["message"] == MESSAGE_END_SESSION:
                self.disconnect()

            return {"status": "ok"}

    @staticmethod
    def send_to_node(address, port, data):
        post_with_retries(
            data=data,
            url=f"http://{address}:{port}/message",
            max_retries=3
        )

    def start(self):
        if check_port(self.address, self.port):
            terminate_process_on_port(self.port)

        uvicorn.run(self.app, host="0.0.0.0", port=self.port)

    def start_training(self, data=None):
        self.fedshare_servers = data["servers"]

        self.round += 1
        cb = TimingCallback()
        self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.set_weights(decode_layer(data["model_weights"]))
        self.model.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=10, callbacks=[cb], verbose=False)
        _, self.current_accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        self.current_training_time = sum(cb.logs)

        for layer in self.model.layers:
            if layer.trainable_weights:
                self.model_shares[layer.name] = [None, None]
                self.scotch_servers_shares[layer.name] = [[], []]

        self.start_secret_sharing()

    def start_secret_sharing(self):
        start_time = timer()
        shares = len(self.fedshare_servers)

        # create shares of model weights
        for layer in self.model.layers:
            if layer.trainable_weights:
                self.model_shares[layer.name] = [None, None]
                self.model_shares[layer.name][0] = list(generate_additive_shares(layer.weights[0], shares))
                self.model_shares[layer.name][1] = list(generate_additive_shares(layer.weights[1], shares))

        self.secret_sharing_time = timer() - start_time

        self.record.append({
            'round': self.round,
            'accuracy': self.current_accuracy,
            'training': self.current_training_time,
            'secret_sharing': self.secret_sharing_time
        })

        # start sending shares to scotch servers
        for server in self.fedshare_servers:
            layer_weights = dict()

            for layer in self.model_shares.keys():
                weight_bias = [None, None]
                weight_bias[0] = self.model_shares[layer][0].pop()
                weight_bias[1] = self.model_shares[layer][1].pop()

                layer_weights[layer] = encode_layer(weight_bias)

            data = {
                "port": self.port,
                "message": MESSAGE_MODEL_SHARE,
                "model_share": layer_weights,
                "data_size": len(self.y_test),
            }

            self.send_to_node(data=data, address=ADDRESS, port=server)

    def accept_shares(self, data):
        start_time = timer()
        for layer in data.keys():
            weight_bias = decode_layer(data[layer])
            self.scotch_servers_shares[layer][0].append(weight_bias[0])
            self.scotch_servers_shares[layer][1].append(weight_bias[1])

        self.secret_sharing_time = self.secret_sharing_time + (timer() - start_time)

    def disconnect(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        output_folder = current_dir + f"/resources/results/{self.client_type}/{self.dataset}"
        os.makedirs(output_folder, exist_ok=True)
        csv_filename = f'client_{self.port - CLIENT_PORT}.csv'
        csv_path = os.path.join(output_folder, csv_filename)
        pd.DataFrame(self.record).to_csv(csv_path, index=False, header=True)
