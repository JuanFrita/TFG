import scipy.stats as stats
import argparse
import os 

def main():   
    
    parser = argparse.ArgumentParser(
        description="Genera el test estadistico entre dos listados.")
    parser.add_argument(
        "output_dir", help="Directorio de guardado")
    parser.add_argument(
        "file_name", help="Nombre del fichero")
    args = parser.parse_args()
    output_dir = args.output_dir
    file_name = args.file_name
    ttest(output_dir, file_name)
    
def ttest(output_dir, file_name):
    
    group_a = [13.4, 6.0, 6.0, 5.8, 6.1, 6.2, 9.4, 5.9, 5.8, 5.8, 5.9, 5.8, 4.2, 5.9, 5.8, 5.8, 6.0, 4.2, 5.9, 
               5.8, 5.9, 6.1, 5.9, 4.3, 5.7, 5.8, 5.9, 5.8, 5.9, 4.2, 5.8, 5.8, 5.9, 5.9, 5.9, 4.4, 5.9, 5.8, 6.0, 
               5.9, 5.8, 4.2, 5.9, 5.8, 5.9, 5.8, 5.8, 4.3, 5.8, 5.9, 5.9, 5.8, 5.9, 4.2, 5.9, 5.9, 5.8, 5.8, 5.9, 
               4.2, 5.8, 5.8, 5.8, 6.0, 5.9, 4.2, 6.0, 5.8, 5.9, 5.9, 6.1, 4.3, 5.8, 5.8, 6.2, 5.9, 5.9, 4.3, 5.8, 
               5.9, 5.8, 5.9, 6.0, 4.3, 5.8, 5.7, 5.8, 5.8, 5.8, 4.2, 5.8, 5.9, 6.0, 5.8, 5.9, 4.4, 5.8, 5.8, 5.9, 
               5.8, 5.8, 4.1, 6.0, 5.9, 5.9, 6.1, 6.1, 4.2, 5.8, 5.9, 5.9, 5.9, 5.9, 4.2, 5.8, 5.9, 6.0, 5.9]

    group_b = [40.09, 31.2, 31.42, 31.01, 31.91, 31.5, 10.13, 32.77, 32.83, 33.13, 9.66, 
           32.99, 32.37, 32.74, 32.41, 32.27, 9.39, 32.63, 33.04, 32.82, 33.21, 32.33, 9.43, 
           32.9, 32.55, 32.52, 32.95, 32.35, 9.38, 32.63, 32.77, 32.55, 32.75, 32.64, 9.27, 
           32.6, 32.53, 32.46, 32.52, 32.44, 9.37, 32.26, 32.47, 33.03, 32.71, 32.51, 9.31, 
           32.7, 32.85, 32.44, 32.38, 32.31, 9.32, 32.56, 32.14, 32.61, 32.16, 32.62, 9.29, 
           32.52, 32.52, 32.34, 32.56, 32.91, 9.25, 32.67, 32.63, 32.43, 32.78, 32.73, 9.43, 
           33.06, 32.56, 32.46, 32.75, 32.43, 9.28, 32.44, 32.48, 32.25, 32.41, 32.05, 9.35, 
           32.45, 32.62, 32.22, 32.37, 32.29, 9.20, 32.42, 32.67, 32.72, 32.76, 32.56, 9.46, 
           32.8, 33.09, 32.43, 32.47, 32.38, 9.32, 32.23, 32.34, 32.35, 32.24, 32.23, 9.33, 
           32.92, 32.53, 32.3, 32.56, 32.59, 9.29, 32.85, 32.42, 32.28, 32.79]
    
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
        
if __name__ == "__main__":
    main()