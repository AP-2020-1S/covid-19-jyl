# Pronóstico de la evolución de casos COVID-19 Colombia.

#### Definición del problema real.

Al final del 2019, el COVID-19 comenzó su propagación a lo largo de China luego de generarse en uno de los mercados de Wuhan, infectando grandes cantidades de gente en una cantidad corta de tiempo. Su mayor afectación es a nivel pulmonar en los humanos, generando fuertes deficiencias que deben controlarse efectivamente antes de que pueda ocasionar la muerte. En la actualidad el brote local se ha controlado efectivamente, mientras que por fuera de la zona se ha propagado rápidamente, logrando que la Organización Mundial de la Saluda la declarara como pandemia en marzo del 2020.

Debido a la afectación global del virus, las agendas políticas de todos los países se volcaron al control y la mitigación de la afectación que este podría ocasionar, debido a esto, y para la toma oportuna de decisiones se vuelve fundamental conocer lo que pasará regionalmente con el virus y anticiparse para proteger a todos los ciudadanos. 
La COVID-19 es una enfermedad respiratoria que se contagia al tener contacto con una persona contagiada, y se transmite vía respiratoria, bucal o por contacto con alguna superficie infectada. Esta forma de transmisión hace muy difícil su control y centra el cuidado de infección en la población con recomendaciones como el lavado constante de manos, el distanciamiento social y el uso constante de tapabocas.


#### Definición del problema de analítica.

El problema se concentra en diseñar un modelo de pronóstico que permita generar una estimación del número de casos activos, infectados, recuperados y muertos para las principales ciudades del país que se producirán en los próximos días (corto y mediano plazo). Este ejercicio produce un reto muy importante al momento de elegir las metodologías adecuadas que permitan acercarse a la curva epidemiológica de la enfermedad, donde al inicio es aceleradamente creciente, luego llega a una estabilización y puede comenzar a decrecer o crecer nuevamente dependiendo de los comportamientos sociales, lo cual representa un reto gigante para cualquier tipo de modelamiento que se intente implementar. Otro de los retos más grandes encontrados es la forma como son reportados los casos, ya que se provoca una escala diferente en el número de casos registrados para los infectados, recuperados y casos fallecidos, además existen situaciones exógenas que no permiten mantener una consistencia coherente durante la serie de tiempo evaluada, los datos no se actualizan de manera inmediata lo cual provoca una alteración en los valores históricos y en los valores pronosticados.

#### Datos Usados.
Para el estudio es usada la fuente de datos abiertos datos.gov (https://www.datos.gov.co/) la cual contiene información reportada desde el 02 de marzo del 2020 para de todos los casos registrados en el territorio nacional.
Los datos que se publican a nivel nacional, se filtran para las cinco principales ciudades de Colombia: Bogotá, Medellín, Cali, Barranquilla y Cartagena, intentando modelar los comportamientos para estas ciudades en cuatro variables: Infectados diarios, recuperados diarios, muertes diarias y casos activos en general.

#### Variables Usadas.

1.	Fecha (reporte, diagnóstico, recuperado, reporte web, muerte)
2.	Ciudad de ubicación
3.	Atención: Infectados diarios, recuperados diarios, muertes diarias y casos activos.
4.	Edad
5.	Sexo

#### Metodologías propuestas.

##### Análisis exploratorio:
Valoración del comportamiento de la serie para conocer cómo se mueve en cada uno de los casos y determinar características de primer interés como componentes de estacionalidad, tendencia o ciclos.

##### Transformación de los datos:
 Es usado el proceso de estandarización.

##### Estrategia para medir generalización: 
Train / test

##### Algoritmo de entrenamiento: 
Modelos supervisados de machine learning.
1.	Modelo de Regresión Líneal
2.	Redes Neuronales (multicapa) estas son evaluadas y ajustadas parametrizando sus parámetros.
3.	Random forest, este es evaluado, ajustado y parametrizando en sus parámetros

##### Media del error para comparar modelos: 
1.	MSE
2.	Accuracy

#### Desarrollo de los modelos.

Para la ejecución de los modelos se definen los siguientes pasos:
1.	Se genera el diseño de una función que permite la obtención de la base de datos de manera diaria (día vencido) la cual es descargada de la fuente de datos ya mencionada. Esta información pasa por diferentes procesos de transformación que permite llevarla a la estructura de datos necesaria producir las bases insumos que serán posteriormente transformadas para el desarrollo de los modelos, las bases generadas son: Infectados_History, Recuperados_History y Muertes_History.

2.	Se diseñan funciones que reciben esta estructura de datos iniciales y de las cuales se procede a conservar, eliminar o construir otras variables para el diseño del pronóstico, esta condición varía de acuerdo al comportamiento de cada tipo de caso evaluado.

3.	Luego se procede con el proceso de partición de los datos: xtrain, ytrain, xtest y ytest y con la estandarización de cada conjunto de datos. La librería usada para este procesamiento es sklear StandardScaler.

4.	Se genera los procesos de entrenamiento haciendo uso del conjunto de datos definidos y se diseñan funciones que permiten reproducir cada modelo y validación de los entrenamientos con las bases de prueba o test. Al final se evalúan tomando como criterios la precisión del modelo y el indicador de error respecto al valor del caso evaluado (MSE).

5.	Construcción de la función de diseño del pronóstico, la cual integra los modelos que bajo el criterio del experto son aptos y se ajustaron mejor para la predicción del caso epidemiológico. Para este caso se usan metodologías que permiten proyectar un tiempo t hacia adelante hasta 10 periodos futuros, o en otros casos se usa metodología vista en clase que hace uso del pronóstico del tiempo t+1 para poder generar el pronóstico del periodo siguiente.

6.	Para la generación de los intervalos de confianza sobre los pronósticos se considera objetivo hacer uso de las desviaciones generadas para diseñar limites superiores e inferiores.

7.	Finalmente se produce el ensamble del proyecto el cual integra los diferentes diseños del modelo seleccionados para el caso de estudio y que permitirá integrar el dashboard y los pronósticos para cada una de las principales ciudades del país.

#### Despliegue de los resultados (producto de datos)
Para esta fase final se busca recrear un producto que logre relacionar indicadores generales del volumen de casos de infectados, recuperados, activos y muertos, concentraciones generales por ciudad y distribución de la edad. Posteriormente se tendrá menú de navegación para consultar de manera independiente cada ciudad junto con las diferentes gráficas que relacionan sus pronósticos individuales. Para el despligue del proyecto es usado Jinja el cual un motor de plantillas con todas las funciones para Python. 


#### Fuentes Web

https://jdvelasq.github.io/courses/ 

https://www.sciencedirect.com/science/article/pii/S2468042720300087 

https://www.datos.gov.co/

https://palletsprojects.com/p/jinja/
