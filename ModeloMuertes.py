from StructureInformation import inputInformation
import pandas as pd
import numpy as np
import seaborn as sns
from ModelRecuperados import PronosticosRecuperados


Data_covid,Infectados_history, Recuperados_history,Muertes_history,Sintomas_history = inputInformation(url="www.datos.gov.co")



Bogota = Sintomas_history[Sintomas_history["ciudad_de_ubicaci_n"] == 'Bogotá D.C.']
Bogota["Ratio"] = Bogota["id_de_caso"]/Bogota['id_de_caso General']
Bogota["Ratio<20"] = Bogota["< 20 Años"]/Bogota['< 20 Años General']
Bogota["Ratio21-40"] = Bogota["21-40 Años"]/Bogota['21-30 Años General']
Bogota["Ratio41-60"] = Bogota["41-60 Años"]/Bogota['41-60 Años General']
Bogota["Ratio61-80"] = Bogota["61-80 Años"]/Bogota['61-80 Años General']
Bogota["Ratio>80"] = Bogota["> 80 Años"]/Bogota['> 80 Años General']






muertos = Data_covid[Data_covid["fecha_de_muerte"].notnull()]
muertos  =muertos[muertos["fis"] != 'Asintomático']

muertos['fis'] = muertos['fis'].str[:10]
muertos['fis'] =pd.to_datetime(muertos['fis'])
muertos['fecha_de_muerte'] = muertos['fecha_de_muerte'].str[:10]
muertos['fecha_de_muerte'] =pd.to_datetime(muertos['fecha_de_muerte'])
muertos['tiempo'] =((muertos['fecha_de_muerte'] - muertos['fis'])/ np.timedelta64(1, 'D')).astype(int)


filterCity=['Bogotá D.C.','Medellín','Cali','Barranquilla','Cartagena de Indias']
muertos = muertos[muertos.ciudad_de_ubicaci_n.isin(filterCity)]


Bogota = muertos[muertos["ciudad_de_ubicaci_n"] == 'Bogotá D.C.']

Bogota = Bogota.groupby(["fecha_de_muerte", "ciudad_de_ubicaci_n"]).count()["id_de_caso"].reset_index()


PronosticosRecuperados(Infectados_history = ,
                       Recuperados_history = ,
                       ciudad,numberLag,predicciones = )


validacio = muertos[['tiempo','fecha_de_muerte','fis']]


a =((muertosostime['fecha_de_muerte'] - muertosostime['fis'])/ np.timedelta64(1, 'D')).astype(int)


sns.distplot(a, hist=True, kde=True,
             bins=int(180/5), color = 'darkblue',
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})



MedeMuertes  =Muertes_history[Muertes_history["ciudad_de_ubicaci_n"] == 'Medellín']
data = MedeMuertes["id_de_caso"].tolist()



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


plt.figure(figsize=(14,5))
plt.plot(data, '.-k')
plt.grid()

# linea vertical para dividir el entrenamiento
# del pronóstico
plt.plot([len(data)-24, len(data)-24], [min(data), max(data)], '--', linewidth=2);




from sklearn.preprocessing import MinMaxScaler

# crea el transformador
scaler = MinMaxScaler()

# escala la serie
data_scaled = scaler.fit_transform( np.array(data).reshape(-1, 1))
# z es un array de listas como efecto
# del escalamiento
data_scaled = [u[0] for u in data_scaled]

plt.figure(figsize=(14,5))
plt.plot(data_scaled, '.-k')
plt.grid()
plt.plot([len(data_scaled)-24, len(data_scaled)-24], [min(data_scaled), max(data_scaled)], '--', linewidth=2);

len(data)


P = 13

X = []
for t in range(P-1, 79-1):
    X.append([data_scaled[t-n] for n in range(P) ])

observed_scaled = data_scaled[P:]

len(X)



from sklearn.neural_network import MLPRegressor

np.random.seed(123456)

H = 1 # Se escoge arbitrariamente

mlp = MLPRegressor(
    hidden_layer_sizes=(H, ),
    activation = 'logistic',
    learning_rate = 'adaptive',
    momentum = 0.0,
    learning_rate_init = 0.1,
    max_iter = 10000)

# Entrenamiento
mlp.fit(
    X[0:215],  # 239 - 24 = 215
    observed_scaled[0:215]
)

# Pronostico
y_scaled_m1 = mlp.predict(X)

plt.figure(figsize=(14,5))
plt.plot(data_scaled, '.-k')
plt.grid()

# No hay pronóstico para los primeros 13 valores
# de la serie
plt.plot([None] * P + y_scaled_m1.tolist(), '-r');

# linea vertical para dividir el entrenamiento
# del pronóstico. Se ubica en el ultimo dato
# usando para entrenamiento
plt.plot([len(data_scaled)-24, len(data_scaled)-24], [min(data_scaled), max(data_scaled)], '--', linewidth=2);