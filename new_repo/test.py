import numpy as np
import os
import argparse
from datetime import datetime
from classes.CNNFactory import CNNFactory


def main():   
    
    parser = argparse.ArgumentParser(
        description="Testea un modelo indicado por parámetro.")
    parser.add_argument(
        "model", help="Modelo a entrenar")
    parser.add_argument("data_origin", help="Nombre de la carpeta con los datos de testing")
    parser.add_argument("output_dir", help="Directorio con los pesos del modelo y donde se guarda el resultado de las predicciones")

    args = parser.parse_args()
    model = args.model
    data_origin = args.data_origin
    output_dir = args.output_dir

    cnnPipeline = CNNFactory.get(model)
    cnnPipeline.runTest(data_origin, output_dir)

if __name__ == "__main__":
    main()
