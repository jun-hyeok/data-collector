import csv
import os
import select
import socket
from concurrent import futures

from pynput import keyboard

from utils import *
from modules.logging import DesktopActivityLogger, PhoneActivityLogger

ISWRITING = True
is_ready = False

# Class to track the status of clients
class Status:
    def __init__(self, clients):
        self.clients = clients
        self.status = {client: "NOT READY" for client in clients}

    def update(self, client, status):
        self.status[client] = status

    def get(self, client):
        return self.status.get(client, "READY")

    @property
    def isallready(self):
        return all(status == "READY" for status in self.status.values())

    def __str__(self):
        ret = ""
        for client, status in self.status.items():
            ret += f"{CLIENTS.get(client, client)}: {status}\n"
        return ret


# Check if a socket is connected
def socket_connected(ip, port):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sc:
        sc.bind((ip, port))
        data, _ = sc.recvfrom(1024)
        if data:
            print(data.decode())
        else:
            raise ValueError("No cognitive")


# Prepare UDP sockets for receiving data from clients
def get_ready(ip_host, port_dict, socket_list):
    for port in port_dict.values():
        sc = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sc.bind((ip_host, port))
        sc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        socket_list.append(sc)


# Function to handle key release events
def on_release(key):
    global ISWRITING
    print("Press Enter to continue...")
    if key == keyboard.Key.enter and not is_ready:
        return False
    if key == keyboard.Key.esc:
        ISWRITING = False
        return False


# Function to concurrently receive data from multiple sockets
def concurrent_recv(sockets, max_worker=20):
    print("Start concurrent")
    workers = min(max_worker, len(sockets))
    with futures.ThreadPoolExecutor(workers) as excutor:
        res = excutor.map(record_data, sockets)
        with keyboard.Listener(on_release=on_release) as listener:
            listener.join()
    return res


# Function to record data from a socket
def record_data(sc):
    global ISWRITING

    while ISWRITING:
        data, (ip, port) = sc.recvfrom(1024)
        client = f"{ip}:{port}"
        sensor = CLIENTS[client]
        print(f"{sensor}: {data.decode()}")
        filename = os.path.join(SAVEDIR, f"{sensor}.csv")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(data.decode().split(","))

    sc.close()
    return True

if __name__ == "__main__":
    clear()
    os.chdir(CURDIR)

    # Read and process configuration
    cfg = load_config(CONFIG)
    ip_host = cfg["IP"]["host"] or DEFAULT.IP.host
    ip_hololens = cfg["IP"]["hololens"] or DEFAULT.IP.hololens
    port_host = cfg["PORT"]["host"] or DEFAULT.PORT.host
    port_hololens = cfg["PORT"]["hololens"] or DEFAULT.PORT.hololens

    # Check socket connection and get ready
    socket_connected(ip_host, port_host)
    PORTS = cfg["PORT"]
    PORTS.pop("host", None)
    PORTS.pop("hololens", None)

    sockets = []
    get_ready(ip_host, PORTS, sockets)

    CLIENTS = {
        f"{ip_hololens}:8001": "widget",
        f"{ip_hololens}:8003": "face",
        f"{ip_hololens}:8004": "imu",
    }
    status = Status(CLIENTS)

    while not is_ready:
        try:
            print(status)
            readable, writable, exceptional = select.select(sockets, [], [])
            for sc in readable:
                data, (ip, port) = sc.recvfrom(1024)
                client = f"{ip}:{port}"

                if "Ready" in data.decode() and status.get(client) != "Ready":
                    status.update(client, "Ready")

                print(status)
                if status.isallready:
                    with keyboard.Listener(on_release=on_release) as listener:
                        listener.join()
                    print("Sending Start signal...")
                    payload = "Start"
                    payload = payload.encode("utf-8").hex()
                    payload = bytes.fromhex(payload)
                    sc.sendto(payload, (ip_hololens, port_hololens))
                    is_ready = True
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            raise SystemExit

    if is_ready:
        print("All devices are ready")
        for sensor in CLIENTS.values():
            filenname = os.path.join(SAVEDIR, f"{sensor}.csv")
            os.makedirs(os.path.dirname(filenname), exist_ok=True)
            f = open(filenname, "w", newline="")
            f.close()
        phone_logger = PhoneActivityLogger(SAVEDIR)
        desktop_logger = DesktopActivityLogger(SAVEDIR)
        phone_logger.start()
        desktop_logger.start()
        concurrent_recv(sockets)