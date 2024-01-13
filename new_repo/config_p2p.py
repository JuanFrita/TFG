import argparse
from classes.p2pnet import P2Pnet

"""
Script que construye el archivo de configuración para entrenar la CNN Bayesian
"""
def main(origen, destino):
    net = P2Pnet()
    net.setFicheros(origen, destino)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Especifica los archivos con el lista de recursos para entrenamiento y testing.")

    parser.add_argument("origen", help="Directorio que contiene las carpetas train y test.")
    parser.add_argument("destino", help="Directorio donde se guardan las listas de entrenamiento y testing.")

    args = parser.parse_args()

    main(args.origen, args.destino)