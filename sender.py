# ------------ system library ------------ #
from socket import *
import sys
from os.path import exists

# ------------ custom library ------------ #
from conf.global_settings import LOG_DIR, DATA_TYPE


class SendServer:
    def __init__(self, IP, PORT):
        self.clientSock = socket(AF_INET, SOCK_STREAM)
        self.clientSock.connect((IP, PORT))
        print(f"Successfully connected to ({IP}:{PORT})")

    def send_file(self, args, global_round, filename):
        file_path = f"{LOG_DIR}/{DATA_TYPE}/{args.net}/global_model/G{str(global_round)}/{filename}"
        msg = filename.encode()
        length = len(msg)
        self.clientSock.sendall(length.to_bytes(4, byteorder="little"))
        self.clientSock.sendall(msg)

        data_transferred = 0

        if not exists(file_path):
            print("No file")
            sys.exit()

        print(f"Sending {filename} ... ")
        with open(file_path, 'rb') as f:
            try:
                data = f.read(1024)
                while data:
                    data_transferred += self.clientSock.send(data)
                    data = f.read(1024)
            except Exception as ex:
                print(ex)

        print(f"Transmission complete {filename} ({data_transferred}/bytes)\n")
