import os
import shutil
import numpy as np
from dotenv import load_dotenv


class P2Pnet:

    """
    Carga las variables de entorno
    """

    def __init__(self):
        load_dotenv()

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
    @staticmethod
    def setCarpetas(imagenes, anotaciones, destino, split_ratio=0.7):

        # no se indica split_ratio
        if split_ratio is None:
            split_ratio = 0.7

        # añadir el origen a la carpeta de imagenes anotaciones y destino
        imagenes = os.path.join(os.getenv('ORIGIN_DIR'), imagenes)
        anotaciones = os.path.join(os.getenv('ORIGIN_DIR'), anotaciones)
        destino = os.path.join(os.getenv('ORIGIN_DIR'), destino)

        # reiniciar las carpetas
        if os.path.exists(destino):
            shutil.rmtree(destino)

        # Obtener la lista de archivos de las carpetas
        files = os.listdir(imagenes)

        # Crear las carpetas train y test si no existen
        train_folder = os.path.join(destino, 'train')
        test_folder = os.path.join(destino, 'test')

        # Mezclar los archivos y dividir en train y test
        np.random.shuffle(files)
        train_files = files[:int(len(files) * split_ratio)]
        test_files = files[int(len(files) * split_ratio):]

        P2Pnet.loadPts(train_files, train_folder, imagenes, anotaciones)
        P2Pnet.loadPts(test_files, test_folder, imagenes, anotaciones)
      
            
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

        #Reiniciar la carpeta de destino
        if os.path.exists(destino):
            shutil.rmtree(destino)
        if not os.path.exists(destino):
                os.makedirs(destino)

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

    """
    Metodos auxiliares para pasar de pts a txt
    """
    @staticmethod
    def read_pts(path):
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

    """
    Método auxiliar para lectura de todos los ficheros de 
    una carpeta y obtener los path completos de cada uno
    """
    def getPaths(origen):
        paths = []
        for file in os.listdir(origen):
            # Usa os.path.join para obtener la ruta absoluta
            paths.append(os.path.abspath(os.path.join(origen, file)))
        return paths
