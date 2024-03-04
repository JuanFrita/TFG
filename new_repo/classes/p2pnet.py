import os
import shutil
import numpy as np
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
import re
import matplotlib.pyplot as plt


class P2Pnet:

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
            Estructura de directorios para la P2Pnet
        """

        # reiniciar las carpetas
        if os.path.exists(destino):
            shutil.rmtree(destino)

        # Crear las carpetas train y test si no existen
        train_folder = os.path.join(destino, 'train')
        test_folder = os.path.join(destino, 'test')

        P2Pnet.loadPts(train_files, train_folder, image_source, anotations_source)
        P2Pnet.loadPts(test_files, test_folder, image_source, anotations_source)

    @staticmethod
    def loadPts(origin, dest,  imagenes, anotaciones):
        scena = 0
        for file in origin:
            string_scena = 'scene0' + \
                str(scena) if scena < 10 else 'scene' + str(scena)

            if not os.path.exists(os.path.join(dest, string_scena)):
                os.makedirs(os.path.join(dest, string_scena))

            # Copiamos tanto la imagen como la anotación correspondiente
            shutil.copy(os.path.join(imagenes, file),
                        os.path.join(dest, string_scena, file))

            # pasar el pts a txt
            annotation_file = os.path.splitext(file)[0] + '.pts'

            txt_file = os.path.splitext(file)[0] + '.txt'

            P2Pnet.pts_to_txt(os.path.join(anotaciones, annotation_file),
                              os.path.join(dest, string_scena, txt_file))

            scena += 1

    @staticmethod
    def setFicheros(origen, destino):
        """
        Genera los ficheros para el entrenamiento y testeo
        origen: nombre de la carpeta que contiene la estructura Train/Test
        destino: fichero donde se guardan la lista de entrenamiento y de test
        """
        train = os.path.join(origen, 'train')
        test = os.path.join(origen, 'test')
        train_destino = os.path.join(destino, 'train.list')
        test_destino = os.path.join(destino, 'test.list')

        P2Pnet.setListFiles(train, train_destino)
        P2Pnet.setListFiles(test, test_destino)

    @staticmethod
    def setListFiles(source, dest):
        escenas = P2Pnet.getPaths(source)
        list_file = open(dest, 'w+')
        for path in escenas:
            ficheros = P2Pnet.getPaths(path)
            img = ficheros[0]
            txt = ficheros[1]
            list_file.write(f'{img} {txt}')
            list_file.write('\n')
        list_file.close()

    @staticmethod
    def read_pts(path):
        """
        Metodos auxiliares para pasar de pts a txt
        """
        with open(path) as f:
            rows = [rows.strip() for rows in f]
        head = rows.index('{') + 1
        tail = rows.index('}')
        raw_points = rows[head:tail]
        coords_set = [point.split() for point in raw_points]
        points = np.array([list([float(point) for point in coords])
                           for coords in coords_set]).astype(np.float32)
        return points

    @staticmethod
    def pts_to_txt(fichero_pts, destino_txt):
        write_file = open(destino_txt, "w+")
        points = P2Pnet.read_pts(fichero_pts)
        for pair in points:  # write the points into a txt
            write_file.write(f'{str(int(pair[0]))} {str(int(pair[1]))}')
            write_file.write('\n')
        write_file.close()

    @staticmethod
    def getPaths(origen):
        """
        Método auxiliar para lectura de todos los ficheros de 
        una carpeta y obtener los path completos de cada uno
        """
        paths = []
        for file in os.listdir(origen):
            # Usa os.path.join para obtener la ruta absoluta
            paths.append(os.path.abspath(os.path.join(origen, file)))
        return paths

    @staticmethod
    def train_model(data_root, epochs, output_dir, checkpoints_dir, tensorboard_dir, batch_size, eval_freq):
        """
        Lanza el comando para entrenar la cnn p2pnet
        """
        # crea los directorios necesarios
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        if os.path.exists(tensorboard_dir):
            shutil.rmtree(tensorboard_dir)
        os.makedirs(tensorboard_dir, exist_ok=True)

        if os.path.exists(checkpoints_dir):
            shutil.rmtree(checkpoints_dir)
        os.makedirs(checkpoints_dir, exist_ok=True)

        comando = f"python ../CrowdCounting-P2PNet-main/train.py --data_root {data_root} --epochs {epochs} --output_dir {output_dir} --checkpoints_dir {checkpoints_dir} --tensorboard_dir {tensorboard_dir} --batch_size {batch_size} --eval_freq {eval_freq} --gpu_id 0"
        print(comando)
        salida, error = P2Pnet.ejecutar_comando(comando)
        if error:
            print("Error:", error.decode())
        else:
            print("Salida:", salida.decode())

    def test_model(weight_path, data_origin, output_dir):

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        comando = f"python ../CrowdCounting-P2PNet-main/run_test_bulk.py --weight_path {weight_path} --data_origin {data_origin} --output_dir {output_dir}"
        print(comando)
        salida, error = P2Pnet.ejecutar_comando(comando)
        if error:
            print("Error:", error.decode())
        else:
            print("Salida:", salida.decode())

    def plot_loss(instance, loss_file, limit_left, limit_right):
        [train_epochs, train_loss] = P2Pnet.plot_individual_loss(loss_file, r"train: loss/loss@(\d+): ([\d.]+)", limit_left, limit_right,  True)
        [val_epochs, val_loss] = P2Pnet.plot_individual_loss(loss_file, r"val: loss/loss@(\d+): ([\d.]+)", limit_left, limit_right)
        plt.plot(train_epochs, train_loss, marker='o', linestyle='-', color='blue', label='Training', markersize=3)
        plt.plot(val_epochs, val_loss, marker='o', linestyle='-', color='red', label='Validation', markersize=3)
        plt.title('Training Vs Validation P2PNet')
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

    def ejecutar_comando(comando):
        "Ejecuta un comando"
        proceso = subprocess.Popen(
            comando, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        salida, error = proceso.communicate()
        return salida, error

    ###############################################
    # FUNCIONES CON ENTRENAMIENTOS PRESTABLECIDOS #
    ###############################################

    def default_train(instance, data_origin):
        fecha_hora_actual = datetime.now()
        P2Pnet.train_model(
            f"../new_repo/assets/data_processed/{data_origin}",
            1000,
            f"../new_repo/assets/results/{data_origin}/{fecha_hora_actual.strftime('%Y-%m-%d_%H-%M-%S')}/output",
            f"../new_repo/assets/results/{data_origin}/{fecha_hora_actual.strftime('%Y-%m-%d_%H-%M-%S')}/checkpoints",
            f"../new_repo/assets/results/{data_origin}/{fecha_hora_actual.strftime('%Y-%m-%d_%H-%M-%S')}/tensorboards",
            6,
            5
        )

    ########################################
    # FUNCIONES CON TESTEOS PRESTABLECIDOS #
    ########################################

    def default_test(instance, weight_path, data_origin):
        """
        Entrenamiento por defecto
        """
        fecha_hora_actual = datetime.now()
        P2Pnet.test_model(
            f"../new_repo/assets/results/{weight_path}/checkpoints/best_mae.pth",
            f"../new_repo/assets/data_processed/{data_origin}/test",
            f"../new_repo/assets/results/{weight_path}/tests/{data_origin}/{fecha_hora_actual.strftime('%Y-%m-%d_%H-%M-%S')}",
        )
