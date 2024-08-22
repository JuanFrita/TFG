import numpy as np
import os
import argparse
from datetime import datetime
from classes.CNNFactory import CNNFactory


def main(imagenes="assets\\images", anotaciones="assets\\annotations", split_ratio=0.8):   
    
    argparse.ArgumentParser(
        description="Especifica la estructura TRAIN/VAL/TEST para un modelo.")
    
    files = os.listdir(imagenes)
    np.random.shuffle(files)
    train_files = files[:int(len(files) * split_ratio)]
    test_val_files = files[int(len(files) * split_ratio):]
    val_files = test_val_files[:int(len(test_val_files) * 0.5)]
    test_files = test_val_files[int(len(test_val_files) * 0.5):]
    
    models = ["bayesian", "p2p"]
    
    for model in models:
        cnnPipeline = CNNFactory.get(model)
        destination = f"assets\\data_processed\\structure{model}{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        cnnPipeline.setupDirectories(imagenes, anotaciones, train_files, val_files, test_files, destination)
        cnnPipeline.setupListFiles(destination, destination)

if __name__ == "__main__":
    main()
