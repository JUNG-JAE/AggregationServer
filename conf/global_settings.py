# ----------- Server parameters ----------- #
DATA_TYPE = "FMNIST"
CHANNEL_SIZE = 1 if DATA_TYPE in ["MNIST", "FMNIST"] else 3

# ----------- System parameters ----------- #
IP = '127.0.0.1'
PORT = 9025
LOG_DIR = "./runs"

CLIENT_ADDR = [
    ('127.0.0.1', 9001),
    ('127.0.0.1', 9002)
]

CLIENT_PER_FILE = 1







