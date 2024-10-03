import time
import threading

from helpers.utils import get_dataset, fetch_index, fetch_dataset
from helpers.constants import SERVER_PORT, NODES, ADDRESS, CLIENT_PORT

from scotch import ScotchNode
from scotch_server import ScotchServer

if __name__ == "__main__":

    DATASET = "mnist"  # str(sys.argv[1])
    SERVERS = 5  # int(sys.argv[2])

    print(f"DATASET: {DATASET}")
    print(f"SERVERS: {SERVERS}")

    indexes = fetch_index(DATASET)
    (X_train, Y_train), (X_test, Y_test) = fetch_dataset(DATASET)
    servers, nodes = [], []
    server_ports, node_ports = [], []
    threads = []

    for i in range(1, SERVERS + 1):
        x_train, y_train, x_test, y_test = get_dataset(
            indexes[i],
            DATASET,
            X_train,
            Y_train,
            X_test,
            Y_test
        )
        server = ScotchServer(
            address=ADDRESS,
            port=SERVER_PORT + i,
            max_nodes=NODES,
            client_type=f'scotch_{SERVERS}',
            dataset=DATASET,
            x_test=x_train,
            y_test=y_train
        )

        server_ports.append(SERVER_PORT + i)
        servers.append(server)

    for i in range(1, NODES + 1):
        x_train, y_train, x_test, y_test = get_dataset(
            indexes[i - 1],
            DATASET,
            X_train,
            Y_train,
            X_test,
            Y_test
        )

        node = ScotchNode(
            address=ADDRESS,
            port=CLIENT_PORT + i,
            client_type=f"scotch_{SERVERS}",
            dataset=DATASET,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test
        )
        node_ports.append(CLIENT_PORT + i)
        nodes.append(node)

    for server in servers:
        time.sleep(2)
        t = threading.Thread(target=server.start, args=(node_ports,))
        t.start()
        threads.append(t)

    for node in nodes:
        time.sleep(2)
        t = threading.Thread(target=node.start, args=(server_ports,))
        t.start()
        threads.append(t)

    for node in nodes:
        time.sleep(2)
        node.start_training()

    for t in threads:
        t.join()
