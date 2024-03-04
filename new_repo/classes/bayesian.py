import os
import shutil
import numpy as np
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import re
from datetime import datetime

class Bayesian:

    @staticmethod
    def setCarpetas(image_source, anotations_source, 
                    train_files, test_files, destino):
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

        #reiniciar las carpetas
        if os.path.exists(destino):
            shutil.rmtree(destino)

        # Crear las carpetas train y test si no existen
        train_folder = os.path.join(destino, 'train')
        test_folder = os.path.join(destino, 'test')
        val_folder = os.path.join(destino, 'val')


        if not os.path.exists(train_folder):
            os.makedirs(train_folder)
        if not os.path.exists(test_folder):
            os.makedirs(test_folder)
        if not os.path.exists(val_folder):
            os.makedirs(val_folder)

        Bayesian.loadPts(train_files, train_folder, image_source, anotations_source)
        Bayesian.loadPts(test_files, test_folder, image_source, anotations_source)
        Bayesian.loadPts(test_files, val_folder, image_source, anotations_source)

            
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
        val_destino = os.path.join(destino, 'val.txt')

        Bayesian.setListFiles(train, train_destino)
        Bayesian.setListFiles(test, test_destino)
        Bayesian.setListFiles(test, val_destino)

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
    
    def plot_loss(instance, loss_file, limit_left, limit_right):
        [train_epochs, train_loss] = Bayesian.plot_individual_loss(loss_file, r"train: loss/loss@(\d+): ([\d.]+)", limit_left, limit_right,  True)
        [val_epochs, val_loss] = Bayesian.plot_individual_loss(loss_file, r"val: loss/loss@(\d+): ([\d.]+)", limit_left, limit_right)
        plt.plot(train_epochs, train_loss, marker='o', linestyle='-', color='blue', label='Training', markersize=3)
        plt.plot(val_epochs, val_loss, marker='o', linestyle='-', color='red', label='Validation', markersize=3)
        plt.title('Training Vs Validation Bayesian')
        plt.xlabel('Época')
        plt.ylabel('Pérdida')
        plt.grid(True)
        plt.legend()
        plt.show()
       
    def plot_individual_loss(loss_file, pattern, limit_left, limit_right, jump=False):
        with open(loss_file, 'r') as archivo:
            datos = archivo.read()
        matches = re.findall(pattern, datos)
        if(jump):
            epochs = [int(match[0]) for match in matches[limit_left:limit_right]if int(match[0]) % 5 == 0]
            losses = [float(match[1]) for match in matches[limit_left:limit_right]if int(match[0]) % 5 == 0]
        else:
            epochs = [int(match[0]) for match in matches[limit_left:limit_right]]
            losses = [float(match[1]) for match in matches[limit_left:limit_right]]
        return [epochs, losses]

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
        comando = f"python ../Bayesian-Crowd-Counting-master/train.py --data-dir {data_root} --save-dir {output_dir}"
        print(comando)
        salida, error = Bayesian.ejecutar_comando(comando)
        if error:
            print("Error:", error.decode())
        else:
            print("Salida:", salida.decode())

    def test_model(data_root, output_dir):

        comando = f"python ../Bayesian-Crowd-Counting-master/test.py --data-dir {data_root} --save-dir {output_dir}"
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
            f"../new_repo/assets/data_processed/{data_origin}",
            f"../new_repo/assets/results/{output_dir}",
        )
        

