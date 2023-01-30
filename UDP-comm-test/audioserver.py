import socket
from io import BytesIO
from datetime import datetime
import threading
import wave
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

STREAM_URL = "127.0.0.1"
BUFFER_SIZE = 1024
PORT = 22222
AUDIO = np.array([])

def get_socket():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) 
    # s.connect((STREAM_URL, PORT)) #TCPIP
    s.bind((STREAM_URL, PORT)) #UDP
    return s

def write_buffer(filename, bytesio: BytesIO):
    '''
    Write buffer data to wav file. 
    '''
    buffer = bytesio.getbuffer()
    with wave.open(filename, "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(48000)
        file.writeframesraw(buffer)
    print(f"[{datetime.now()}] Wrote buffer. Size: {buffer.nbytes} bytes")

def drawLive(): #ToDo:revision
    global AUDIO
    plt.ion()
    fig, ax = plt.subplots()
    fig.canvas.set_window_title("Audio")
    ax.set_title("Audio Streaming")
    while True:
        try:
            if len(AUDIO) > 0:
                time = np.linspace(start=0, stop=len(AUDIO)/48000, num = len(AUDIO))
                ax.plot(time, AUDIO)
                plt.show()
                plt.pause(0.00001)
            else:
                pass
        except KeyboardInterrupt:
            break

def main():
    global AUDIO
    s = get_socket()
    # th = threading.Thread(target=drawLive, daemon=True)
    # th.start()
    with BytesIO() as bIO:
        try:
            while True:
                data, _ = s.recvfrom(BUFFER_SIZE)
                if data != None:
                    bIO.write(data)
                    tmp = np.frombuffer(data, dtype=np.int16) / 32767
                    AUDIO = np.concatenate((AUDIO, tmp))

                if (len(AUDIO) / 48000 > 10):
                    break
        except KeyboardInterrupt: return
        finally: 
            write_buffer("stream.wav", bytesio=bIO)
            print("end")

if __name__ == "__main__":
    main()