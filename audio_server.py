import os
import socket
import wave
from datetime import datetime
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
from utils import *

AUDIO = np.array([])

def get_socket(ip, port):
    sc = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # s.connect((IP, PORT)) #TCPIP
    sc.bind((ip, port))  # UDP
    return sc


def write_buffer(filename, bytesio: BytesIO):
    buffer = bytesio.getbuffer()
    with wave.open(filename, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(48000)
        f.writeframesraw(buffer)
    print(f"[{datetime.now()}] Wrote buffer. Size: {buffer.nbytes} bytes")


# def draw_live():  # ToDo:revision
#     global AUDIO
#     plt.ion()
#     fig, ax = plt.subplots()
#     fig.canvas.set_window_title("Audio")
#     ax.set_title("Audio Streaming")
#     while True:
#         try:
#             if len(AUDIO) > 0:
#                 time = np.linspace(start=0, stop=len(AUDIO) / 48000, num=len(AUDIO))
#                 ax.plot(time, AUDIO)
#                 plt.show()
#                 plt.pause(0.00001)
#             else:
#                 pass
#         except KeyboardInterrupt:
#             break


def main(sc, buffer_size):
    global AUDIO
    # th = threading.Thread(target=drawLive, daemon=True)
    # th.start()
    with BytesIO() as bIO:
        try:
            while True:
                data, _ = sc.recvfrom(buffer_size)
                if data != None:
                    bIO.write(data)
                    tmp = np.frombuffer(data, dtype=np.int16) / 32767
                    AUDIO = np.concatenate((AUDIO, tmp))

                if len(AUDIO) / 48000 > 10:
                    break
        except KeyboardInterrupt:
            return
        finally:
            filenname = os.path.join(SAVEDIR, "stream.wav")
            write_buffer(filenname, bytesio=bIO)
            print("end")

if __name__ == "__main__":
    clear()
    os.chdir(CURDIR)

    # Read and process configuration
    cfg = load_config(CONFIG)
    ip = cfg["AUDIO"]["ip"] or DEFAULT.AUDIO.ip
    port = cfg["AUDIO"]["port"] or DEFAULT.AUDIO.port
    buffer_size = cfg["AUDIO"]["buffer_size"] or DEFAULT.AUDIO.buffer_size

    # Check socket connection and get ready
    sc = get_socket(ip, port)
    main(sc, buffer_size)
