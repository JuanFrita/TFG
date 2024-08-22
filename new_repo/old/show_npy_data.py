import argparse
from new_repo.classes.old.bayesian import Bayesian

def main(map):
    bayesian = Bayesian()
    bayesian.ShowNpyHotMap(map)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Especifica la ruta del fichero .npy de la red bayesian.")
    parser.add_argument("map", help="Fichero.npy")
    args = parser.parse_args()
    main(args.map)