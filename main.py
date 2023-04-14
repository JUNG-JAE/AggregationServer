# ------------ system library ------------ #
import argparse
import glob
import os
import time

# ------------ custom library ------------ #
from system_utility import set_global_round
from learning_utility import get_network, load_model, save_model,aggregation
from receiver import run_server
from sender import SendServer
from conf.global_settings import IP, PORT, CLIENT_ADDR, LOG_DIR, DATA_TYPE

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    args = parser.parse_args()

    global_round = set_global_round(args)
    
    run_server(args, global_round)

    local_models = [get_network(args) for _ in range(len(CLIENT_ADDR))]

    for i in range(len(CLIENT_ADDR)):
        local_models[i] = load_model(local_models[i], args, global_round, f"shard{i + 1}")

    print("Global model aggregation\n")
    aggregation_model = aggregation(args, *local_models)
    save_model(aggregation_model, args, global_round)

    model_files = glob.glob(f"{LOG_DIR}/{DATA_TYPE}/{args.net}/global_model/G{str(global_round)}/*.pt")

    time.sleep(1)

    for ip, port in CLIENT_ADDR:
        for file in model_files:
            sender = SendServer(ip, port)
            sender.send_file(args, global_round, os.path.basename(file))
            sender.clientSock.close()


