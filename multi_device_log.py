from http import server
from socket import *
import time
import csv
import threading
from datetime import datetime
import sys
import msvcrt
from pynput import mouse
from pynput import keyboard
from datetime import datetime
import time
import csv
import threading
import sys

state_phone = "ready"
state_pc = 0
IPADDRESS = "192.168.1.129"

class UdpComms():
    def __init__(self,udpIP,portTX,portRX,enableRX=False,suppressWarnings=True):
        """
        Class for UDP communication. 
        Source :  Two-way communication between Python 3 and Unity (C#) - Y. T. Elashry (https://github.com/Siliconifier/Python-Unity-Socket-Communication)
        Constructor
        :param udpIP: Must be string e.g. "127.0.0.1"
        :param portTX: integer number e.g. 8000. Port to transmit from i.e From Python to other application
        :param portRX: integer number e.g. 8001. Port to receive on i.e. From other application to Python
        :param enableRX: When False you may only send from Python and not receive. If set to True a thread is created to enable receiving of data
        :param suppressWarnings: Stop printing warnings if not connected to other application
        """

        self.udpIP = udpIP
        self.udpSendPort = portTX
        self.udpRcvPort = portRX
        self.enableRX = enableRX
        self.suppressWarnings = suppressWarnings # when true warnings are suppressed
        self.isDataReceived = False
        self.dataRX = None

        # Connect via UDP
        self.udpSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # internet protocol, udp (DGRAM) socket
        self.udpSock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # allows the address/port to be reused immediately instead of it being stuck in the TIME_WAIT state waiting for late packets to arrive.
        self.udpSock.bind((udpIP, portRX))

        # Create Receiving thread if required
        if enableRX:
            self.rxThread = threading.Thread(target=self.ReadUdpThreadFunc, daemon=True)
            self.rxThread.start()

    def __del__(self):
        self.CloseSocket()

    def CloseSocket(self):
        # Function to close socket
        self.udpSock.close()

    def SendData(self, strToSend):
        # Use this function to send string to C#
        self.udpSock.sendto(bytes(strToSend,'utf-8'), (self.udpIP, self.udpSendPort))

    def ReceiveData(self):
        """
        Should not be called by user
        Function BLOCKS until data is returned from C#. It then attempts to convert it to string and returns on successful conversion.
        An warning/error is raised if:
            - Warning: Not connected to C# application yet. Warning can be suppressed by setting suppressWarning=True in constructor
            - Error: If data receiving procedure or conversion to string goes wrong
            - Error: If user attempts to use this without enabling RX
        :return: returns None on failure or the received string on success
        """
        if not self.enableRX: # if RX is not enabled, raise error
            raise ValueError("Attempting to receive data without enabling this setting. Ensure this is enabled from the constructor")

        data = None
        try:
            data, _ = self.udpSock.recvfrom(1024)
            data = data.decode('utf-8')
        except WindowsError as e:
            if e.winerror == 10054: # An error occurs if you try to receive before connecting to other application
                if not self.suppressWarnings:
                    print("Are You connected to the other application? Connect to it!")
                else:
                    pass
            else:
                raise ValueError("Unexpected Error. Are you sure that the received data can be converted to a string")

        return data

    def ReadUdpThreadFunc(self): # Should be called from thread
        """
        This function should be called from a thread [Done automatically via constructor]
                (import threading -> e.g. udpReceiveThread = threading.Thread(target=self.ReadUdpNonBlocking, daemon=True))
        This function keeps looping through the BLOCKING ReceiveData function and sets self.dataRX when data is received and sets received flag
        This function runs in the background and updates class variables to read data later

        """

        self.isDataReceived = False # Initially nothing received

        while True:
            data = self.ReceiveData()  # Blocks (in thread) until data is returned (OR MAYBE UNTIL SOME TIMEOUT AS WELL)
            self.dataRX = data # Populate AFTER new data is received
            self.isDataReceived = True
            # When it reaches here, data received is available

    def ReadReceivedData(self):
        """
        This is the function that should be used to read received data
        Checks if data has been received SINCE LAST CALL, if so it returns the received string and sets flag to False (to avoid re-reading received data)
        data is None if nothing has been received
        :return:
        """

        data = None

        if self.isDataReceived: # if data has been received
            self.isDataReceived = False
            data = self.dataRX
            self.dataRX = None # Empty receive buffer

        return data

class saveLog(threading.Thread):
    '''
    Write smartphone using log to csv file. 
    '''
    def __init__(self, writer):
        threading.Thread.__init__(self)
        self.writer = writer

    def run(self):
        # The event listener will be running in this block
        global state_phone
        cur_time = 0 #ToDo : Get device time. 
        while(True):
            if state_phone == "using":
                sw = 1
            else:
                sw = 0
            self.writer.writerow([cur_time, str(sw)])
            cur_time += 0.333
            time.sleep(0.333)
            if state_phone == "End":
                break
            if msvcrt.kbhit() and msvcrt.getch() == chr(27).encode():
                    break

class phoneLog(threading.Thread):
    '''
    Get phone log through UDP network communication. 
    '''
    def __init__(self, pid):
        threading.Thread.__init__(self)
        self.pid = pid
    
    def run(self):
        global state_phone
        server_socket = socket(AF_INET, SOCK_STREAM)

        ip = IPADDRESS
        port = 8001
        server_socket.bind((ip, port))
        server_socket.listen(100)
        trial_num = self.pid
        file_path = datetime.now().strftime('%Y-%m-%d')+"_"+trial_num+ "phone" +".csv"
        open_csv = open(file_path, "w", newline='')
        writer = csv.writer(open_csv)
        logtracker = saveLog(writer)
        logtracker.start()

        while True:
            server_socket.settimeout(1)
            print(state_phone)
            if msvcrt.kbhit() and msvcrt.getch() == chr(27).encode():
                open_csv.close()
                break
            try:
                client_socket, addr = server_socket.accept()
            except timeout:
                continue
            server_socket.settimeout(None)
            state_phone = client_socket.recv(1024).decode('utf-8')
            if state_phone == "End":
                open_csv.close()
                break

class detectMouse(threading.Thread):
    '''
    Detect whether mouse cursor is moving or not
    '''
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        # The event listener will be running in this block
        global state_pc
        while(True):
            with mouse.Events() as events:
                # Block at most one second
                event = events.get(1.0)
                if event is None:
                    print('You did not interact with the mouse within one second')
                    state_pc = 0
                else:
                    print('Received event Mouse {}'.format(event))
                    state_pc = 1

                if msvcrt.kbhit() and msvcrt.getch() == chr(27).encode():
                    break

class detectKeyboard(threading.Thread):
    '''
    Detect whether keyboard is being pressed or not
    '''
    def __init__(self, path):
        threading.Thread.__init__(self)
        self.path = path

    def run(self):
        # The event listener will be running in this block
        global state_pc
        while(True):
            with keyboard.Events() as events:
                # Block at most one second
                event = events.get(1.0)
                if event is None:
                    print('You did not interact with the keyboard within one second')
                    state_pc = 0 
                elif str(event) == "Press(key=Key.esc)": # press ESC key to EXIT
                    self.path.close()
                    state_pc = 3
                    print("escaped")
                    break
                else:
                    print(event)
                    print('Received event keyboard {}'.format(event))
                    state_pc  = 1

class desktopLog(threading.Thread):
    '''
    Write keyboard and mouse log to csv
    '''
    def __init__(self, pid):
        threading.Thread.__init__(self)
        self.pid = pid

    def run(self):
        global state_pc
        trial_num = self.pid
        file_path = datetime.now().strftime('%Y-%m-%d')+"_"+trial_num+ "desktop" +".csv"
        open_csv = open(file_path, "w", newline='')
        writer = csv.writer(open_csv)
        mouse_ = detectMouse()
        keyboard_ = detectKeyboard(open_csv)
        mouse_.start()
        keyboard_.start()
        cur_time = 0
        while(True):
            if state_pc == 3:
                print("save")
                break
            # open_csv.write(datetime.now().strftime('%Y-%m-%d_%H_%M_%S') + "," + str(state_pc) + "/n")
            writer.writerow([cur_time, str(state_pc)])
            cur_time += 0.333
            time.sleep(0.333)

if __name__ == '__main__':
    s = input() # Get participants id. 
    phoneLogging = phoneLog(s)
    desktopLogging = desktopLog(s)
    phoneLogging.start()
    desktopLogging.start()