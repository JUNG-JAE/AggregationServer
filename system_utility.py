# ----------- System library ----------- #
import os
import logging
import pathlib

# ----------- Custom library ----------- #
from conf.global_settings import LOG_DIR, DATA_TYPE


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def set_global_round(args):
    global_round_path = pathlib.Path(LOG_DIR) / DATA_TYPE / args.net / "global_model"

    if not global_round_path.exists():
        print("[ ==================== Global Round: 1 ==================== ]")
        global_round = 1
    else:
        rounds = [int(p.name[1:]) for p in global_round_path.glob("G*")]
        global_round = max(rounds) + 1
        print(f"[ ==================== Global Round: {global_round:2} ==================== ]")

    new_round_path = global_round_path / f"G{global_round}"
    try:
        new_round_path.mkdir(parents=True, exist_ok=True)
    except OSError:
        print(f"Error: Creating global model {global_round:2} directory")

    return global_round




