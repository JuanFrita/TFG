import numpy as np
import os
import argparse
from datetime import datetime
from classes.CNNFactory import CNNFactory


def main(imagenes="assets\\images", anotaciones="assets\\annotations", split_ratio=0.7):   
    
    parser = argparse.ArgumentParser(
        description="Entrena un modelo indicado por parámetro.")
    parser.add_argument(
        "model", help="Modelo a entrenar")
    parser.add_argument("data_origin", help="Nombre de la carpeta con los datos de entrenamiento y validación")

    args = parser.parse_args()
    model = args.model
    data_origin = args.data_origin

    cnnPipeline = CNNFactory.get(model)
    cnnPipeline.runTrain(data_origin)

if __name__ == "__main__":
    main()
