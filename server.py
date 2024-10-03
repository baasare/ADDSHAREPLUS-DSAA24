import os
import time
import json
import uvicorn
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI
from timeit import default_timer as timer
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding

from helpers.utils import post_with_retries, encode_layer, decode_layer, get_lenet5_classification, get_private_key
from helpers.utils import check_port, terminate_process_on_port, get_dataset, combine_csv_files, NumpyDecoder

from helpers.constants import MESSAGE_END_SESSION, MESSAGE_START_ASSEMBLY, MESSAGE_FL_UPDATE
from helpers.constants import MESSAGE_TRAINING_COMPLETED, MESSAGE_START_SECRET_SHARING, ADDRESS
from helpers.constants import MESSAGE_SHARING_COMPLETE, ROUNDS, MESSAGE_START_TRAINING, MESSAGE_FL_UPDATE_ENCRYPTED


class Server:
    def __init__(self, server_id, address, port, max_nodes, client_type, dataset, indexes, x_train, y_train, x_test,
                 y_test):
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

        self.private_key = get_private_key('server')

        @self.app.post("/message")
        def message(data: dict):
            print(f"SERVER RECEIVED: {data['message']} from PORT: {data['port']}")

            if data["message"] in [MESSAGE_FL_UPDATE, MESSAGE_FL_UPDATE_ENCRYPTED]:
                self.fl_update(data["port"], data["model_weights"], data["message"])

            elif data["message"] == MESSAGE_TRAINING_COMPLETED:
                self.start_secret_sharing()

            elif data["message"] == MESSAGE_SHARING_COMPLETE:
                self.start_assembly(data["port"])

            return {"status": "ok"}

    def start(self):
        if check_port(self.address, self.port):
            terminate_process_on_port(self.port)

        self.check_current_round()
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

    def check_current_round(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        output_folder = current_dir + f"/resources/results/{self.client_type}/{self.dataset}"
        csv_filename = 'server.csv'
        csv_path = os.path.join(output_folder, csv_filename)

        if os.path.exists(csv_path):
            server_records = pd.read_csv(csv_path)
            self.round = server_records['round'].max()
            self.record = server_records.to_dict('records')
        else:
            self.round = 0

    def start_round(self, nodes=None):
        if nodes:
            self.nodes = nodes

        print(f'Starting round ({self.round + 1})')

        self.start_time = timer()
        self.pending_nodes = self.nodes.copy()
        for layer in self.global_model.layers:
            if layer.trainable_weights:
                self.average_weights[layer.name] = [[], []]

        data = {
            "port": "SERVER",
            "nodes": self.nodes,
            "message": MESSAGE_START_TRAINING,
            "model_architecture": self.global_model.to_json(),
            "model_weights": encode_layer(self.global_model.get_weights()),
        }
        self.send_to_node(data)

    def fl_update(self, node, data, message):
        if message == MESSAGE_FL_UPDATE_ENCRYPTED:
            encrypted_messages = decode_layer(data)
            decrypted_messages = []

            for chunk in encrypted_messages:
                decrypted_messages.append(
                    self.private_key.decrypt(
                        chunk,
                        padding.OAEP(
                            mgf=padding.MGF1(algorithm=hashes.SHA256()),
                            algorithm=hashes.SHA256(),
                            label=None
                        )
                    )
                )

            separator = b''
            decoded_data = separator.join(decrypted_messages).decode('utf8')
            data = json.loads(decoded_data, cls=NumpyDecoder)

            for layer in data.keys():
                temp_weight = data[layer]

                if len(self.average_weights[layer][0]) == 0 and len(self.average_weights[layer][1]) == 0:
                    self.average_weights[layer][0] = temp_weight[0] / len(self.nodes)
                    self.average_weights[layer][1] = temp_weight[1] / len(self.nodes)
                else:
                    self.average_weights[layer][0] += temp_weight[0] / len(self.nodes)
                    self.average_weights[layer][1] += temp_weight[1] / len(self.nodes)

        else:
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
        self.global_model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
                                  loss='categorical_crossentropy', metrics=['accuracy'])
        _, self.current_accuracy = self.global_model.evaluate(self.X_test, self.y_test, verbose=0)
        self.end_time = timer() - self.start_time
        print('Accuracy: ', self.current_accuracy)
        print(f'Round ({self.round + 1}) Time: {self.end_time}')

        self.record.append({
            'round': self.round + 1,
            'accuracy': self.current_accuracy,
            'fl': self.end_time,
        })

        current_dir = os.path.dirname(os.path.realpath(__file__))
        output_folder = current_dir + f"/resources/results/{self.client_type}/{self.dataset}"
        os.makedirs(output_folder, exist_ok=True)
        csv_filename = 'server.csv'
        csv_path = os.path.join(output_folder, csv_filename)
        pd.DataFrame(self.record).to_csv(csv_path, index=False, header=True)

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

        current_dir = os.path.dirname(os.path.realpath(__file__))
        output_folder = current_dir + f"/resources/results/{self.client_type}/{self.dataset}"
        os.makedirs(output_folder, exist_ok=True)
        csv_filename = 'server.csv'
        csv_path = os.path.join(output_folder, csv_filename)
        pd.DataFrame(self.record).to_csv(csv_path, index=False, header=True)

        self.send_to_node(data)
        combine_csv_files(f"{self.client_type}", f"{self.dataset}")
        terminate_process_on_port(self.port)

    def start_assembly(self, port):
        data = {
            "port": "SERVER",
            "message": MESSAGE_START_ASSEMBLY,
        }
        self.send_to_node(data, port=port)

    def start_secret_sharing(self):
        self.training_completed_count += 1
        if self.training_completed_count == len(self.nodes):
            self.training_completed_count = 0
            data = {
                "port": "SERVER",
                "message": MESSAGE_START_SECRET_SHARING,
            }
            self.send_to_node(data)
