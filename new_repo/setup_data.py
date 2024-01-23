from classes.p2pnet import P2Pnet
from classes.bayesian import Bayesian
from datetime import datetime
from pathlib import Path


"""
Obtener por parametros los directorios para crear la estructura para la cnn Bayesian
"""
def main(imagenes="assets\\images", anotaciones="assets\\annotations", split_ratio=0.7):   
    fecha_hora_actual = datetime.now()
    netP2p = P2Pnet()
    destinoP2P = f"assets\\data_processed\\estructuraP2P{fecha_hora_actual.strftime('%Y-%m-%d_%H-%M-%S')}"
    netP2p.setCarpetas(imagenes, anotaciones, destinoP2P, split_ratio)
    netP2p.setFicheros(destinoP2P, destinoP2P)
    
    netBayesian = Bayesian()
    destinoBayesian =  f"assets\\data_processed\\estructuraBayesian{fecha_hora_actual.strftime('%Y-%m-%d_%H-%M-%S')}"
    netBayesian.setCarpetas(imagenes, anotaciones, destinoBayesian, split_ratio)
    netBayesian.setFicheros(destinoBayesian, destinoBayesian)
    netBayesian.preprocessData(destinoBayesian)

if __name__ == "__main__":
    main()
