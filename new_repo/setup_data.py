from classes.p2pnet import P2Pnet
from classes.bayesian import Bayesian
from datetime import datetime
from pathlib import Path
import numpy as np
import os


"""
Obtener por parametros los directorios para crear la estructura para la cnn Bayesian
"""
def main(imagenes="assets\\images", anotaciones="assets\\annotations", split_ratio=0.7):   
    ##obtener los ficheros de entrenamiento y de testing
    files = os.listdir(imagenes)
    np.random.shuffle(files)
    train_files = files[:int(len(files) * split_ratio)]
    test_files = files[int(len(files) * split_ratio):]
    
    fecha_hora_actual = datetime.now()
    #montar la estructura necesaria para la red p2p
    netP2p = P2Pnet()
    destinoP2P = f"assets\\data_processed\\estructuraP2P{fecha_hora_actual.strftime('%Y-%m-%d_%H-%M-%S')}"
    netP2p.setCarpetas(imagenes, anotaciones, train_files, test_files, destinoP2P)
    netP2p.setFicheros(destinoP2P, destinoP2P)
    
    #montar la estructura necesaria para la red bayesian
    netBayesian = Bayesian()
    destinoBayesian =  f"assets\\data_processed\\estructuraBayesian{fecha_hora_actual.strftime('%Y-%m-%d_%H-%M-%S')}"
    netBayesian.setCarpetas(imagenes, anotaciones, train_files, test_files, destinoBayesian)
    netBayesian.setFicheros(destinoBayesian, destinoBayesian)
    netBayesian.preprocessData(destinoBayesian)

if __name__ == "__main__":
    main()
