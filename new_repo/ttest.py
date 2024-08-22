import scipy.stats as stats
import argparse

def main():   
    
    parser = argparse.ArgumentParser(
        description="Entrena un modelo indicado por parámetro.")
    parser.add_argument(
        "output_dir", help="Directorio de guardado")

    ttest()
    
def ttest():
    group_a = [85, 90, 92, 88, 87, 84, 91, 89, 86, 87]
    group_b = [78, 82, 80, 88, 86, 79, 83, 81, 85, 84]

    # Realizamos una prueba t de dos muestras independientes
    t_stat, p_value = stats.ttest_ind(group_a, group_b)

    # Establecemos un nivel de significancia (alfa)
    alfa = 0.05

    # Comprobamos si rechazamos la hipótesis nula
    if p_value < alfa:
        print("Hay una diferencia significativa entre los grupos.")
    else:
        print("No hay evidencia de una diferencia significativa entre los grupos.")