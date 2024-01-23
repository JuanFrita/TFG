import argparse
from classes.p2pnet import P2Pnet
from datetime import datetime

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Especifica la estructura TRAIN/TEST para la red P2P.")
    parser.add_argument(
        "config", help="Configuración de entrenamiento a cargar")
    parser.add_argument("data_origin", help="Nombre de la carpeta a cargar")
    args = parser.parse_args()
    funcion = getattr(P2Pnet(), args.config)
    funcion(args.data_origin)
