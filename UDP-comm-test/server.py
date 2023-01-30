import socket
import select
from pynput import keyboard
import os
from concurrent import futures
import csv
import time
import inquirer
import yaml
from easydict import EasyDict as edict
import threading

ISWRITING = True
SENSORLIST = ['cognitive', 'imu', 'face', 'widget']
PORTS = {10025 : "widget", 
            10026 : "face", 
            10027 : "imu"}

def load_config(path):
    '''
    config.yaml에서 ip 주소를 받아온다. 
    '''
    with open(path) as fin:
        config = edict(yaml.safe_load(fin))

    return config

def on_release(key):
    global ISWRITING
    print("Press ENTER to continue")
    if key == keyboard.Key.enter:
        # Stop listener
        if not isReady:
            return False
    if key == keyboard.Key.esc:
        ISWRITING = False
        return False

def clear():
    os.system( 'cls' )

def print_status(ip_dict, status):
    clear()
    i = 0
    for k, v in ip_dict.items():
        print("%s is %s"%(v, status[i]))
        i+=1

def create_csv():
    for sensor in SENSORLIST:
        f = open('%s.csv'%sensor, 'w', newline='')
        f.close()

def record_data(udp_socket):
    '''
    하나의 디바이스에서 받은 데이터를 csv로 저장
    '''
    global ISWRITING
    client_ip_dict = {"%s:8001"%cfg.hololens_ip:"widget", 
    "%s:8003"%cfg.hololens_ip:"face", 
    "%s:8004"%cfg.hololens_ip:"imu"} 
    while True: 
        client_data, client = udp_socket.recvfrom(1024)
        client_ip, client_port = client[0], client[1]
        sensor = client_ip_dict[client_ip + ":" + str(client_port)]
        print("[%s] %s"%(sensor, client_data.decode()))
        # time.sleep(0.03)
        with open("%s.csv"%sensor, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(client_data.decode().split(","))
        if not ISWRITING:
            udp_socket.close()
            break
    
    return True
        

def concurrent_recv(udp_sockets ,max_worker=20):
    '''
    각 디바이스에 대해서 record_data를 실행
    '''
    print("Start concurrent")
    workers = min(max_worker, len(udp_sockets))
    with futures.ThreadPoolExecutor(workers) as excutor:
        res = excutor.map(record_data, udp_sockets) #run at thread....
        with keyboard.Listener(on_release=on_release) as listener:
            listener.join()
    return res


if __name__ == "__main__":
    clear()
    cfg = load_config('config.yaml')
    isReady = False
    IP_ADDRESS = cfg.main_ip

    #Cognitive Stage
    cognitive_socket = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    cognitive_socket.bind((IP_ADDRESS, 10024))

    data, _ = cognitive_socket.recvfrom(1024)
    if data != None:
        print(data.decode())
        cognitive_socket.close()
    else:
        raise "No cognitive"

    #Device Ready Stage
    udp_sockets = []
    for port in list(PORTS.keys()):
        server_socket = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        server_socket.bind((IP_ADDRESS, port))
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        udp_sockets.append(server_socket)

    client_ip_dict = {"%s:8001"%cfg.hololens_ip:"widget", 
    "%s:8003"%cfg.hololens_ip:"face", 
    "%s:8004"%cfg.hololens_ip:"imu"}

    status = []
    for _ in range(len(client_ip_dict)):
        status.append("NOT READY")
    
    while not isReady:
        try:
            print_status(client_ip_dict, status)
            readable, writable, exceptional = select.select(udp_sockets,[],[])
            for socket in readable:
                (client_data,(client_ip,client_port)) = socket.recvfrom(1024)

                if "Ready" in client_data.decode() and status[list(client_ip_dict.keys()).index(client_ip + ":" + str(client_port))] != "Ready":
                    # 각 디바이스에서 "Ready" 메세지를 받으면 status 업데이트
                    idx = list(client_ip_dict.keys()).index(client_ip + ":" + str(client_port))
                    status[idx] = "Ready"

                print_status(client_ip_dict, status)
                    
                if "NOT READY" not in status:
                    # 모든 디바이스에 "Ready"를 받으면 "Start" 메시지를 각 디바이스 (지금은 홀로렌즈에만 다른 디바이스에는 보낼 필요없을수도) 에 전송하고 기록을 시작. 
                    print("Press ENTER to continue")
                    with keyboard.Listener(on_release=on_release) as listener:
                        listener.join()
                    print("Send Start")
                    payload_string = "Start"
                    payload_ascii = payload_string.encode("utf-8").hex()
                    payload = bytes.fromhex(payload_ascii)
                    socket.sendto(payload, ((cfg.hololens_ip,8001)))
                    isReady = True
        except KeyboardInterrupt:
            break
            
    if isReady:
        create_csv()
        concurrent_recv(udp_sockets)
    