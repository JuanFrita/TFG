import argparse
from classes.p2pnet import P2Pnet

"""
Obtener por parametros los directorios para crear la estructura para la cnn Bayesian
"""
def main(imagenes, anotaciones, destino, split_ratio=0.7):    
    net = P2Pnet()
    net.setCarpetas(imagenes, anotaciones, destino, split_ratio)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Especifica la estructura TRAIN/TEST para la red P2P.")

    parser.add_argument("imagenes", help="Directorio que contiene las imagenes")
    parser.add_argument("anotaciones", help="Directorio que contiene las anotaciones")
    parser.add_argument("destino", help="Destino de la estructura Train/Test")
    parser.add_argument("split_ratio", help="Ratio del conjunto de entrenamiento y testing")

    args = parser.parse_args()

    main(args.imagenes, args.anotaciones, args.destino, float(args.split_ratio))