# TFG

## Descripción del proyecto
Monitorizar automáticamente la ocupación de las playas de las Illes Balears.

**enlace del proyecto base** https://github.com/ZhihengCV/Bayesian-Crowd-Counting
**enlace del proyecto base** https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet?tab=readme-ov-fileg


https://arxiv.org/pdf/2107.12746.pdf

https://arxiv.org/pdf/1908.03684.pdf

**citación especial** Ma, Z., Wei, X., Hong, X., & Gong, Y. (2019). Bayesian loss for crowd count estimation with point supervision. In Proceedings of the IEEE International Conference on Computer Vision (pp. 6142-6151).


Buenas tardes Pedro y Antoni. 

Os mando este correo para que tengáis constancia de los avances que se van haciendo en el TFG y que dudas me han surgido. La semana pasada no pude avanzar mucho por eso os solicité posponer a principios de la semana que viene la reunión para que sea mas sustancial.

Ya tengo preparada en una clase propia las funciones para lanzar los entrenamientos y testeos de la red p2p, me falta hacer lo propio para la bayesian (sigue el mismo estilo así que es bastante rápido de montar). 

También investigando los parámetros del script de testeo de la red p2p, vi que había un parámetro para loguear en un archivo,la pérdida por epoch y otro para ir marcando los checkpoints de validación cada x epochs, entonces he creado otro script que coge el fichero de salida y con un regex saca la pérdida por epoch para poder hacer la visualización de la gráfica. 

Sobre todo esto es con el fin de poder facilitar los experimentos ya que los modelos tienen bastantes parámetros que sería interesante ir ajustando, y también por que me sirve para poder comparar el rendimientos de los modelos que es el punto más importante de esta semana de trabajo. 

Sobre comparar y medir el rendimiento de los modelos es donde me han surgido más dudas. 

Os pongo un contexto de como entiendo que funciona cada modelo por si os interesa:

El modelo Bayesian está basado en un mapa de densidad pero con la función de pérdida bayesiana. De lo que he entendido es que un mapa de densidad convencional pondría un valor de 0 a los pixeles donde no hay una persona (cabeza) y un valor de 1 donde haya, en el caso de este modelo cada pixel marca la probabilidad de que haya una persona en esa ubicación. Como consecuencia el conteo total de personas es la suma total de las probabilidades, y en el paper indican que este enfoque ayuda a que el modelo sea bueno generalizando.

El modelo P2P sigue un enfoque basado únicamente en puntos, es decir, predecir la posición exacta donde se ubica una persona (cabeza). Proponen una nueva métrica ( Density Normalized Average Precision ) muy interesante para no solo basarnos en el MAE y MSE del conteo total, si no verificar que la localización de los puntos y la performance del conteo son correctas. 

Entonces el MAE y MSE seguro que se usarán para medir el rendimiento. El modelo P2P si que ofrece una forma de ver si el conteo es realmente acertado o simplemente da bien "por suerte". En el caso del modelo Bayesian puede que se complique un poco más de hecho en su paper usan el MAE y MSE, pero ciertamente habrá que buscar una forma de poder medir que el mapa de densidad generado es correcto. 