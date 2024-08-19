import argparse
from classes.p2pnet import P2Pnet
from datetime import datetime
from classes.CNNFactory import CNNFactory

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Muestra la comparativa de la pérdida de la validación frente al entrenamiento.")
    parser.add_argument("model", help="Nombre del modelo")
    parser.add_argument("loss_file", help="Nombre de la carpeta a cargar")
    parser.add_argument("limit_left", help="Max epoch", type=int)
    parser.add_argument("limit_right", help="Max epoch", type=int)
    args = parser.parse_args()
    
    args = parser.parse_args()
    model = args.model

    cnnPipeline = CNNFactory.get(model)
    cnnPipeline.plotTrainVsValLoss(model, args.loss_file, args.limit_left, args.limit_right)
