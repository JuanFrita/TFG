import os

directorio = 'C:/Users/Usuario/TFG/resources/Test_Training_bayesian/test_trained_full_1/test'  # Reemplaza con la ruta a tu directorio
archivo_salida = 'test.txt'  # Nombre del archivo de salida

nombres_archivos = []
for nombre_archivo in os.listdir(directorio):
    nombre, extension = os.path.splitext(nombre_archivo)
    if extension == '.jpg':
        nombres_archivos.append(nombre)

nombres_no_repetidos = list(set(nombres_archivos))
nombres_con_extension = [nombre + '.jpg' for nombre in nombres_no_repetidos]

with open(archivo_salida, 'w') as archivo:
    archivo.write('\n'.join(nombres_con_extension))
    archivo.write('\n')

print('Archivo guardado exitosamente:', archivo_salida)