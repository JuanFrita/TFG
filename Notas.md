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

C:/Users/Usuario/TFG/resources/test_trained_full_1/

C:/Users/Usuario/TFG/resources/Test_Training_bayesian/test_trained_full_1

python preprocess_dataset.py --origin-dir C:/Users/Usuario/TFG/resources/Test_Training_bayesian/test_trained_full_1 --data-dir   C:/Users/Usuario/TFG/resources/Test_Training_bayesian/test_trained_full_1


python test.py --data-dir C:/Users/juanf/OneDrive/Escritorio/TFG/resources/Test_Training/test_pretrained --save-dir C:/Users/juanf/OneDrive/Escritorio/TFG/resources/Test_Training/test_pretrained/results

python train.py --data_dir C:/Users/juanf/OneDrive/Escritorio/TFG/resources/Test_Training/test_trained --save_dir C:/Users/juanf/OneDrive/Escritorio/TFG/resources/Test_Training/results

 cd Bayesian-Crowd-Counting-master

Para generar los puntos de el Bayesian Crowd Counting hay que usar el tipo de fichero dlib.pts

Prueba para mejorar la calidad de imagen 
https://www.youtube.com/watch?v=nPnQm7HFWJs

Para mejorar toda la imagen
https://www.youtube.com/watch?v=d-CPvHkltXA
https://www.youtube.com/watch?v=nQF6UjoUbMA
https://github.com/bycloudai/Real-ESRGAN-Windows

Aqui sale como usar lasrx de amd con pytorch
https://github.com/RadeonOpenCompute/ROCm/issues/1698

Para entrenar el modelo se usará un pc con una rtx 3060ti.

Propuesta de mejor modelo (redactar en mi cuaderno un poco todos los avances para la reunión)
https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet

Recuerda hacerte un entorno para p2p net y para bayesina por que necesitan versiones de torch diferentes

para tirar el entrenamiento de la p2pnet
CUDA_VISIBLE_DEVICES=0 python train.py --data_root C:\\Users\\Usuario\\TFG\\resources\\Test_Training_p2p\\test_pretrained  --epochs 2000     --lr_drop 3500     --output_dir C:\\Users\\Usuario\\TFG\\logs     --checkpoints_dir ./weights     --tensorboard_dir ./logs     --lr 0.0001     --lr_backbone 0.00001     --batch_size 1     --eval_freq 300     --gpu_id 0