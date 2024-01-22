import argparse
from classes.bayesian import Bayesian

def main(map):
    bayesian = Bayesian()
    bayesian.ShowNpyHotMap(map)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Especifica la estructura TRAIN/TEST para la red P2P.")
    parser.add_argument("map", help="Fichero.npy")
    args = parser.parse_args()
    main(args.map)