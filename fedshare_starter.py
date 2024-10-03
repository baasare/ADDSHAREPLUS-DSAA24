import sys
import time
import threading

from fedshare import FedShareNode
from fedshare_server import FedShareServer
from fedshare_leadserver import FedShareLeadServer

from helpers.utils import fetch_index, fetch_dataset, get_dataset
from helpers.constants import ADDRESS, SERVER_PORT, NODES, CLIENT_PORT

if __name__ == "__main__":

    DATASET = str(sys.argv[1])
    SERVERS = int(sys.argv[2])

    print(f"DATASET: {DATASET}")
    print(f"SERVERS: {SERVERS}")

    indexes = fetch_index(DATASET)
    (x_train, y_train), (x_test, y_test) = fetch_dataset(DATASET)

    servers, nodes = [], []
    server_ports, node_ports = [], []
    threads = []

    # lead server
    lead_server = FedShareLeadServer(
        address=ADDRESS,
        port=SERVER_PORT,
        client_type=f'fedshare_{SERVERS}',
        dataset=DATASET,
        indexes=indexes,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test
    )

    # creating servers
    for i in range(1, SERVERS + 1):
        server = FedShareServer(
            address=ADDRESS,
            port=SERVER_PORT + i,
            max_nodes=NODES,
            client_type=f'fedshare_{SERVERS}',
            dataset=DATASET,
            indexes=indexes,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test
        )

        server_ports.append(SERVER_PORT + i)
        servers.append(server)

    # creating nodes
    for i in range(1, NODES + 1):
        X_train, Y_train, X_test, Y_test = get_dataset(
            indexes[i - 1],
            DATASET,
            x_train,
            y_train,
            x_test,
            y_test
        )

        node = FedShareNode(
            address=ADDRESS,
            port=CLIENT_PORT + i,
            client_type=f"fedshare_{SERVERS}",
            dataset=DATASET,
            x_train=X_train,
            y_train=Y_train,
            x_test=X_test,
            y_test=Y_test
        )
        node_ports.append(CLIENT_PORT + i)
        nodes.append(node)

    # lead server
    lead_server_thread = threading.Thread(target=lead_server.start)
    lead_server_thread.start()
    threads.append(lead_server_thread)

    for server in servers:
        time.sleep(2)
        t = threading.Thread(target=server.start)
        t.start()
        threads.append(t)

    for node in nodes:
        time.sleep(2)
        t = threading.Thread(target=node.start)
        t.start()
        threads.append(t)

    lead_server.start_round(server_ports, node_ports)

    for t in threads:
        t.join()
