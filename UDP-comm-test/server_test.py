import socket
import select
'''
디바이스 IP 주소를 확인하고 싶을때 사용. server 안에 합쳐서 실행 전에 무조건 실행 되도록 해도 좋을듯. 
'''
if __name__ == "__main__":
    # constant of program
    isReady = False
    IP_ADDRESS = "192.168.1.141"
    # IP_ADDRESS = "127.0.0.1"
    PORT = 3001

    # the payload that UDP server will response
    PAYLOAD_STRING = f"I got your message\r\n"
    payload_ascii = PAYLOAD_STRING.encode("utf-8").hex()
    payload = bytes.fromhex(payload_ascii)

    # sockets that listen to ports 10024 until 10123, total of 100 ports
    udp_sockets = []
    server_socket = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    server_socket.bind((IP_ADDRESS, PORT))
    udp_sockets.append(server_socket)

    while True:
        readable, writable, exceptional = select.select(udp_sockets,[],[])
        for socket in readable:
            (client_data,(client_ip,client_port)) = socket.recvfrom(1024)
            print(client_data.decode())
            print(client_ip)
            print(client_port)