import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets, linear_model, metrics
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
 ####################################################################### analisis de correlacion#######

a =((Data_covid['fecha_recuperado'] - Data_covid['fecha_diagnostico'])/ np.timedelta64(1, 'D')).dropna().astype(int)


sns.distplot(a, hist=True, kde=True,
             bins=int(180/5), color = 'darkblue',
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})

###################### modelo Regresion rezagos###############


def ModelData(DataX,DataY,ciudad,numberLag):

    DataX_city = DataX[DataX["ciudad_de_ubicaci_n"] == ciudad]
    DataY_city = DataY[DataY["ciudad_de_ubicaci_n"] == ciudad]

    infectados = DataX_city[["fecha_diagnostico", "id_de_caso"]]
    infectados.columns = ['Fecha', 'Numero_de_Infectados']
    recuperado = DataY_city[["fecha_recuperado", "id_de_caso"]]
    recuperado.columns = ['Fecha', 'Numero_de_Recuperados']
    dataModel = pd.merge(infectados, recuperado, on='Fecha', how="outer")

    dataModel["Numero_de_Infectados"] = dataModel["Numero_de_Infectados"].fillna(0)
    dataModel["Numero_de_Recuperados"] = dataModel["Numero_de_Recuperados"].fillna(0)

    dataModel = dataModel.sort_values(by=['Fecha'], ascending=True)
    X_Value = dataModel[["Numero_de_Infectados"]]
    Rezagos = pd.concat([dataModel, X_Value.shift()], axis=1)
    for x in range(2, numberLag):
        Rezagos = pd.concat([Rezagos, X_Value.shift(x)], axis=1)

    Rezagos = Rezagos.sort_values(by=['Fecha'], ascending=False)
    Rezagos = Rezagos.dropna()

    namescolumns = dataModel.columns.tolist() + ["Rezago" + str(x) for x in range(1, Rezagos.shape[1] - 2)]
    Rezagos.columns = namescolumns
    return Rezagos

def error_cuadratico(y_real,y_pred):
    return np.sum((y_real - y_pred)**2/len(y_real))

Data_Modelo = ModelData(DataX = Infectados_history, DataY = Recuperados_history, ciudad = "Medellín",numberLag = 50)

Variables_Independientes = ["Rezago" + str(x) for x in range(1, Data_Modelo.shape[1] - 2)]
Variable_Dependiente = ['Numero_de_Recuperados']



X = Data_Modelo[Variables_Independientes]
Y = Data_Modelo[Variable_Dependiente]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,
                                                    random_state=1)





modelo = linear_model.LinearRegression().fit(X_train, y_train)


scores = cross_validate(modelo, X_train, y_train, cv=3,
                        scoring= 'neg_mean_squared_error')

scores.get("test_score")


Data_Error = pd.DataFrame(columns=["Rezago","error"])

for x in Variables_Independientes:
    X_train1 = X_train[[x]]
    X_test1 = X_test[[x]]
    modelo = linear_model.LinearRegression()
    modelo.fit(X_train1, y_train)
    y_pred = modelo.predict(X_test1)
    error = error_cuadratico(y_real=y_test, y_pred=y_pred)
    value = pd.DataFrame({"Rezago":x,
                          "error":error})
    Data_Error = Data_Error.append(value,ignore_index=False)

Data_Error = Data_Error.sort_values(by=['error'], ascending=True)




X_trainReza18 = X_train[["Rezago18","Rezago24"]]
X_test1 = X_test[["Rezago18","Rezago24"]]
modelo18 = linear_model.LinearRegression()
modelo18.fit(X_trainReza18, y_train)
y_pred = modelo18.predict(X_test1)

error = error_cuadratico(y_real=y_test, y_pred=y_pred)


df = pd.DataFrame('Yreal':y_test,'ypred':y_pred)


