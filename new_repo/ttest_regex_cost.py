import scipy.stats as stats
import argparse
import os 
import re
import numpy as np

def main():   
    
    parser = argparse.ArgumentParser(
        description="Genera el test estadistico entre dos listados.")
    parser.add_argument(
        "log_file_1", help="Fichero de log con los costes")
    parser.add_argument(
        "log_file_2", help="Fichero de log con los costes")
    parser.add_argument(
        "output_dir", help="Directorio de guardado")
    parser.add_argument(
        "file_name", help="Nombre del fichero")
    args = parser.parse_args()
    output_dir = args.output_dir
    file_name = args.file_name
    log_file_1 = args.log_file_1
    log_file_2 = args.log_file_2
    ttest(log_file_1, log_file_2, output_dir, file_name)
    
def ttest(log_file_1, log_file_2, output_dir, file_name):
    
    group_a = find_matches(log_file_1)
    print(np.mean(group_a))
    group_b = find_matches(log_file_2)
    print(np.mean(group_b))
    
    # Realizamos una prueba t de dos muestras independientes
    t_stat, p_value = stats.ttest_ind(group_a, group_b)

    # Establecemos un nivel de significancia (alfa)
    alfa = 0.05

    result = "Rechazamos la hipótesis nula" if p_value < alfa else "No rechazamos la hipótesis nula"

    # Define la ruta del archivo .txt
    file_path = os.path.join(output_dir, file_name)
    
    print(result)
    
    # Escribe los resultados en el archivo .txt
    with open(file_path, "w") as archivo:
        archivo.write(f"Resultados de la prueba t de dos muestras independientes:\n")
        archivo.write(f"t-stat: {t_stat}\n")
        archivo.write(f"p-value: {p_value}\n")
        archivo.write(f"Resultado: {result}\n")


def find_matches(file):
    with open(file, 'r') as archivo:
        datos = archivo.read()
    matches = re.findall(r"Cost ([\d.]+) sec", datos)
    costs = [float(valor) for valor in matches]
    return costs

if __name__ == "__main__":
    main()