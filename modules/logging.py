import csv
import msvcrt
import os
import socket
import threading
import time
from . import tracking
from .tracking import KeboardTracker, MouseTracker
from . import IPADDRESS, STATE_PHONE

# Thread to write smartphone usage log to a CSV file
class PhoneActivityTracker(threading.Thread):
    '''
    Write smartphone usage log to a CSV file.
    '''
    def __init__(self, writer):
        super().__init__()
        self.writer = writer

    def run(self):
        global STATE_PHONE
        cur_time = 0  # ToDo: Get device time.
        while True:
            if STATE_PHONE == "using":
                sw = 1
            else:
                sw = 0
            self.writer.writerow([cur_time, str(sw)])
            cur_time += 0.333
            time.sleep(0.333)
            if STATE_PHONE == "End":
                break
            if msvcrt.kbhit() and msvcrt.getch() == chr(27).encode():
                break

# Thread to receive and write phone log through UDP network communication
class PhoneActivityLogger(threading.Thread):
    '''
    Receive and write phone log through UDP network communication.
    '''
    def __init__(self, savedir):
        super().__init__()
        self.path = savedir
    
    def run(self):
        global STATE_PHONE
        sc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        ip = IPADDRESS
        port = 8001
        sc.bind((ip, port))
        sc.listen(100)
        filename = os.path.join(self.path, "phone.csv")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w", newline='') as f:
            writer = csv.writer(f)
            log_tracker = PhoneActivityTracker(writer)
            log_tracker.start()

            while True:
                sc.settimeout(1)
                print(STATE_PHONE)
                if msvcrt.kbhit() and msvcrt.getch() == chr(27).encode():
                    f.close()
                    break
                try:
                    client_socket, _ = sc.accept()
                except socket.timeout:
                    continue
                sc.settimeout(None)
                STATE_PHONE = client_socket.recv(1024).decode('utf-8')
                if STATE_PHONE == "End":
                    f.close()
                    break

# Thread to write keyboard and mouse log to a CSV file
class DesktopActivityLogger(threading.Thread):
    '''
    Write keyboard and mouse log to a CSV file.
    '''
    def __init__(self, savedir):
        super().__init__()
        self.path = savedir

    def run(self):
        filename = os.path.join(self.path, "desktop.csv")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w", newline='') as f:
            writer = csv.writer(f)
            mouse_thread = MouseTracker()
            keyboard_thread = KeboardTracker(f)
            mouse_thread.start()
            keyboard_thread.start()
            cur_time = 0
            while True:
                if tracking.STATE_PC == 3:
                    print("save")
                    break
                writer.writerow([cur_time, str(tracking.STATE_PC)])
                cur_time += 0.333
                time.sleep(0.333)