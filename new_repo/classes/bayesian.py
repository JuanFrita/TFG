import os
import shutil
import numpy as np
from dotenv import load_dotenv 

class Bayesian:

    def __init__(self):
        """
        Carga las variables de entorno
        """
        load_dotenv()
        

    @staticmethod
    def setCarpetas(imagenes, anotaciones, destino, split_ratio=0.7):
        """
        Genera la estructura de los ficheros train y test

        Args:
            imagenes: ruta con las imágenes
            anotaciones: ruta con las anotaciones .pts
            destino: nombre de la carpeta donde se guardará la estructuras
            split_ratio: ratio para el conjunto de entrenamiento y testing

        Returns:
            Estructura de directorios para la CNN Bayesian +
        """

        #no se indica split_ratio
        if split_ratio is None:
            split_ratio = 0.7 
    
        #añadir el origen a la carpeta de imagenes anotaciones y destino
        imagenes = os.path.join(os.getenv('ORIGIN_DIR'), imagenes)
        anotaciones = os.path.join(os.getenv('ORIGIN_DIR'), anotaciones)
        destino = os.path.join(os.getenv('ORIGIN_DIR'), destino)

        #reiniciar las carpetas
        shutil.rmtree(destino)

        # Obtener la lista de archivos de las carpetas (solo necesitamos una lista porque los nombres deben coincidir)
        files = os.listdir(imagenes)

        # Crear las carpetas train y test si no existen
        train_folder = os.path.join(destino, 'train')
        test_folder = os.path.join(destino, 'test')

        if not os.path.exists(train_folder):
            os.makedirs(train_folder)
        if not os.path.exists(test_folder):
            os.makedirs(test_folder)

        # Mezclar los archivos y dividir en train y test
        np.random.shuffle(files)
        train_files = files[:int(len(files) * split_ratio)]
        test_files = files[int(len(files) * split_ratio):]
        
        Bayesian.loadPts(train_files, train_folder, imagenes, anotaciones)
        Bayesian.loadPts(test_files, test_folder, imagenes, anotaciones)
            
    @staticmethod
    def loadPts(origin, dest, imagenes, anotaciones):
          for file in origin:
            # Copiamos tanto la imagen como la anotación correspondiente
            shutil.copy(os.path.join(imagenes, file), dest)
            annotation_file = os.path.splitext(
                file)[0] + '.pts'  # cambiar la extensión a .pts
            shutil.copy(os.path.join(
                anotaciones, annotation_file), dest)

    @staticmethod
    def setFicheros(origen):
        """
        Genera los ficheros para el entrenamiento y testeo
        origen: nombre de la carpeta que contiene la estructura Train/Test
        """
        

