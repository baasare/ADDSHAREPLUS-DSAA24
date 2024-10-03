import os
import sys
import time
import uvicorn
import threading
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI

from helpers import constants
from area_x_server import AreaXAddsharePlusServer
from helpers.utils import check_port, terminate_process_on_port, decode_layer, TimingCallback
from helpers.utils import get_area_x_dataset, get_dataset, post_with_retries, encode_layer


class AreaXFedAvg:

    def __init__(self, address, port, client_type, dataset, x_train, y_train, x_test, y_test):
        self.app = FastAPI()
        self.port = port
        self.address = address

        self.dataset = dataset
        self.client_type = client_type

        self.model = None
        self.epochs = constants.EPOCHS

        self.record = list()
        self.current_training_time = 0
        self.round, self.mae, self.rmse, self.mape = 0, 0, 0, 0

        self.X_train, self.y_train, self.X_test, self.y_test = x_train, y_train, x_test, y_test

        @self.app.post("/message")
        def message(data: dict):
            print(f"PORT {self.port} RECEIVED: {data['message']}")
            if data["message"] == constants.MESSAGE_START_TRAINING:
                self.start_training(data)

            elif data["message"] == constants.MESSAGE_END_SESSION:
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
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            loss=tf.keras.losses.mae,
            metrics=[tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanAbsolutePercentageError()]
        )
        self.model.set_weights(decode_layer(global_model["model_weights"]))

        cb = TimingCallback()
        self.model.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=10, callbacks=[cb], verbose=False)
        self.mae, self.rmse, self.mape = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        self.current_training_time = sum(cb.logs)

        model_weights = dict()
        for layer in self.model.layers:
            if layer.trainable_weights:
                model_weights[layer.name] = encode_layer(layer.get_weights())

        self.record.append({
            'round': self.round,
            'loss': self.mae,
            'rmse': self.rmse,
            'mape': self.mape,
            'training': self.current_training_time,
        })

        data = {
            "port": self.port,
            "message": constants.MESSAGE_FL_UPDATE,
            "model_weights": model_weights,
        }

        self.send_to_node(constants.ADDRESS, constants.SERVER_PORT, data)

    def end_session(self, data):
        model_weights = decode_layer(data['model_weights'])
        self.model.set_weights(model_weights)
        self.disconnect()

    def disconnect(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        output_folder = current_dir + f"/resources/results/{self.client_type}/{self.dataset}"
        os.makedirs(output_folder, exist_ok=True)
        csv_filename = f'client_{self.port - constants.CLIENT_PORT}.csv'
        csv_path = os.path.join(output_folder, csv_filename)
        pd.DataFrame(self.record).to_csv(csv_path, index=False, header=True)


if __name__ == "__main__":

    DATASET = "area_x"  # str(sys.argv[1])
    print(f"DATASET: {DATASET}")

    x, y, _, _ = get_area_x_dataset(9)

    nodes = []
    ports = []
    threads = []

    server = AreaXAddsharePlusServer(
        server_id=constants.SERVER_ID,
        address=constants.ADDRESS,
        port=constants.SERVER_PORT,
        max_nodes=constants.NODES,
        pruning_type=constants.MAGNITUDE,
        client_type='fed_avg',
        dataset=DATASET,
        x=x,
        y=y
    )
    server_thread = threading.Thread(target=server.start)

    for i in constants.FIELDS:
        X_train, Y_train, X_test, Y_test = get_area_x_dataset(i)

        node = AreaXFedAvg(
            address=constants.ADDRESS,
            port=constants.CLIENT_PORT + int(i),
            client_type="fed_avg",
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
