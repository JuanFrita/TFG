# Notas
Estos son notas sobre los posibles avances y siguientes pasos del proyecto

Importar los modelos con los pesos ya entrenados:
    - UCF-QNRF
    - ShanghaiTech A
    - ShanghaiTech B

Reentrenar el modelo con mis imágenes
Probar con diferentes preprocesamiento de las imáganes:
    - Igualar el brillo a la que más tiene (para aprovechar las que son de tarde).
    - Pasar a escala de grises (no creo que funcione).
    - Quitar bloom.
    - Enfatizar los detalles de la foto.

Notas: 
    - Se ha modificado el archivo preprocess_dataset para tomar las anotaciones desde ficheros 
    .pts.

Primeras pruebas: test_pretrained (contiene los resultados con la neurona preentrenada y las imágenes en crudo)

Con el modelo preentrenado : 156 test y 150 reales

python preprocess_dataset.py --origin-dir D:/TFG/resources/Test_Training/test_pretrained --data-dir D:/TFG/resources/Test_Training/test_pretrained
python test.py --data-dir D:/TFG/resources/Test_Training/test_pretrained --save-dir D:/TFG/resources/Test_Training/test_pretrained/results

python preprocess_dataset.py --origin-dir C:/Users/juanf/OneDrive/Escritorio/TFG/resources/Test_Training/test_pretrained --data-dir   C:/Users/juanf/OneDrive/Escritorio/TFG/resources/Test_Training/test_pretrained

python test.py --data-dir C:/Users/juanf/OneDrive/Escritorio/TFG/resources/Test_Training/test_pretrained --save-dir C:/Users/juanf/OneDrive/Escritorio/TFG/resources/Test_Training/test_pretrained/results

Para generar los puntos de el Bayesian Crowd Counting hay que usar el tipo de fichero dlib.pts

Aqui sale como usar lasrx de amd con pytorch
https://github.com/RadeonOpenCompute/ROCm/issues/1698

Para entrenar el modelo se usará un pc con una rtx 3060ti.