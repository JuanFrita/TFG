import argparse
from classes.p2pnet import P2Pnet
from datetime import datetime

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Especifica la estructura TRAIN/TEST para la red P2P.")
    parser.add_argument("data_origin", help="Nombre de la carpeta a cargar")
    parser.add_argument("limit_left", help="Max epoch", type=int)
    parser.add_argument("limit_right", help="Max epoch", type=int)
    args = parser.parse_args()
    P2Pnet().plot_loss(f"assets/results/{args.data_origin}/output/run_log.txt", args.limit_left , args.limit_right)
