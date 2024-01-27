import os
import shutil
import numpy as np
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime

class Bayesian:

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
            
        #reiniciar las carpetas
        if os.path.exists(destino):
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
    def setFicheros(origen, destino):
        """
        Genera los ficheros para el entrenamiento y testeo
        origen: nombre de la carpeta que contiene la estructura Train/Test
        destino: fichero donde se guardan la lista de entrenamiento y de test
        """
        train = os.path.join(origen, 'train')
        test = os.path.join(origen, 'test')
        train_destino = os.path.join(destino, 'train.txt')
        test_destino = os.path.join(destino, 'test.txt')

        Bayesian.setListFiles(train, train_destino)
        Bayesian.setListFiles(test, test_destino)
    
    @staticmethod
    def setListFiles(source, dest):
        ficheros = Bayesian.getPaths(source)
        list_file = open(dest, 'w+')
        for fichero in ficheros: 
            if fichero.endswith(".jpg"):
                list_file.write(f'{fichero}')
                list_file.write('\n')
        list_file.close()
        
    @staticmethod
    def preprocessData(destino):
        "Ejecuta el comando de la cnn bayesian"
        comando = f"python ../Bayesian-Crowd-Counting-master/preprocess_dataset.py --origin-dir ../new_repo/{destino} --data-dir ../new_repo/{destino}" 
        print(comando)
        salida, error = Bayesian.ejecutar_comando(comando)
        if error:
            print("Error:", error.decode())
        else:
            print("Salida:", salida.decode())

    @staticmethod
    def ejecutar_comando(comando):
        "Ejecuta un comando"
        proceso = subprocess.Popen(comando, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        salida, error = proceso.communicate()
        return salida, error
    
    @staticmethod
    def getPaths(origen):
        """
        Método auxiliar para lectura de todos los ficheros de 
        una carpeta y obtener los path completos de cada uno
        """
        paths = []
        for file in os.listdir(origen):
            # Usa os.path.join para obtener la ruta absoluta
            paths.append(file)
        return paths
    
    @staticmethod
    def ShowNpyHotMap(imagen):
        npy_file_path = imagen
        jpg_file_path = npy_file_path.replace('.npy', '.jpg')
        points = np.load(npy_file_path)
        img = Image.open(jpg_file_path)
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.scatter(points[:, 0], points[:, 1], c='red', marker='o', s=2)
        plt.show()
        
    @staticmethod
    def train_model(data_root, output_dir):
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            os.makedirs(output_dir, exist_ok=True)
        comando = f"python ../Bayesian-Crowd-Counting-master/train.py --data_dir {data_root} --save_dir {output_dir}"
        print(comando)
        salida, error = Bayesian.ejecutar_comando(comando)
        if error:
            print("Error:", error.decode())
        else:
            print("Salida:", salida.decode())

    def test_model(data_root, output_dir):
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            os.makedirs(output_dir, exist_ok=True)
        comando = f"python ../Bayesian-Crowd-Counting-master/test.py --data_dir {data_root} --save_dir {output_dir}"
        print(comando)
        salida, error = Bayesian.ejecutar_comando(comando)
        if error:
            print("Error:", error.decode())
        else:
            print("Salida:", salida.decode())
    
    ###############################################
    # FUNCIONES CON ENTRENAMIENTOS PRESTABLECIDOS #
    ###############################################
     
    @staticmethod
    def default_train(instance, data_origin):
        fecha_hora_actual = datetime.now()
        Bayesian.train_model(
            f"../new_repo/assets/data_processed/{data_origin}",
            f"../new_repo/assets/results/{data_origin}/{fecha_hora_actual.strftime('%Y-%m-%d_%H-%M-%S')}/output",
        )
        
    ########################################
    # FUNCIONES CON TESTEOS PRESTABLECIDOS #
    ########################################
    
    def default_test(instance, data_origin, output_dir):
        Bayesian.test_model(
            f"../new_repo/assets/data_processed/{data_origin}/test",
            f"../new_repo/assets/results/{output_dir}/output",
        )
        

