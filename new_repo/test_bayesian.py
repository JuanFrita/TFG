import argparse
from classes.bayesian import Bayesian
from datetime import datetime

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Especifica la estructura TRAIN/TEST para la red bayesian.")
    parser.add_argument(
        "config", help="Configuración de entrenamiento a cargar")
    parser.add_argument("data_origin", help="Nombre de la carpeta a cargar")
    parser.add_argument("output_dir", help="Nombre de la carpeta que contiene el modelo")
    args = parser.parse_args()
    funcion = getattr(Bayesian(), args.config)
    funcion(args.data_origin, args.output_dir)
