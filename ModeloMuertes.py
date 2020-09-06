from StructureInformation import inputInformation
from sklearn import  linear_model
import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pylab as plot
from DescriptiveAndPlots import grafica_series_muerte


def error_cuadratico(estimador, x,y_real):
    y_pred = estimador.predict(x)
    return np.sum((y_real - y_pred)**2/len(y_real))

def BaseModeloMuertes(Sintomas_history,Muertes_history,ciudad):

    x = Sintomas_history[Sintomas_history["ciudad_de_ubicaci_n"] == ciudad]
    datex = x[["fis"]]

    x["Ratio"] = x["id_de_caso"] / x['id_de_caso General']
    x["Ratio<20"] = x["< 20 Años"] / x['< 20 Años General']
    x["Ratio21-40"] = x["21-40 Años"] / x['21-30 Años General']
    x["Ratio41-60"] = x["41-60 Años"] / x['41-60 Años General']
    x["Ratio61-80"] = x["61-80 Años"] / x['61-80 Años General']
    x["Ratio>80"] = x["> 80 Años"] / x['> 80 Años General']

    x = x[['Ratio<20',
           'Ratio21-40', 'Ratio41-60', 'Ratio61-80', 'Ratio>80']]
    x = x.fillna(0)

    pesosedad = x.mean(axis=0, skipna=True)
    y = Muertes_history[Muertes_history["ciudad_de_ubicaci_n"] == ciudad]
    y = y[['fecha_de_muerte', 'id_de_caso']]
    y.columns = ['fecha', 'Muertes']
    x_ponder = Sintomas_history[Sintomas_history["ciudad_de_ubicaci_n"] == ciudad]
    matrix_ponde = x_ponder[['21-40 Años', '41-60 Años', '61-80 Años', '< 20 Años', '> 80 Años']]
    pesosedad.index = matrix_ponde.columns
    x_ponder = matrix_ponde.mul(pesosedad, axis=1)
    X = pd.concat([datex, x_ponder], axis=1)
    X = X.fillna(0)
    X.columns = ['fecha', '21-40 Años', '41-60 Años', '61-80 Años', '< 20 Años', '> 80 Años']
    X = pd.merge(X, y, on=['fecha'], how="left")
    X = X.dropna()
    fechamax = y[['fecha']].max()
    return X, fechamax

def TrainMuertes(DataModelMuertes,predicciones,iniciogenerador):
    x = DataModelMuertes[['21-40 Años', '41-60 Años', '61-80 Años', '< 20 Años', '> 80 Años']]
    y = DataModelMuertes[['Muertes']]
    rowsdata = x.shape[0] - predicciones
    inicio = rowsdata - iniciogenerador

    Data_Error = pd.DataFrame(columns=["Rezago", "error_Prediccion", "iteraccion"])
    for rezago in range(10, 19):
        for i in range(inicio, rowsdata):
            Xtrain = x.iloc[0:i]
            Ytrain = y.iloc[rezago:i + rezago]
            Xtest = x.iloc[i:i + predicciones]
            Ytest = y.iloc[i + predicciones:i + predicciones + predicciones]
            if Xtest.shape[0] == Ytest.shape[0]:
                modelo = linear_model.LinearRegression()
                modelo = modelo.fit(Xtrain, Ytrain)
                error = error_cuadratico(estimador=modelo, x=Xtest, y_real=Ytest)
                value = pd.DataFrame({"Rezago": rezago,
                                      "error_Prediccion": error, "iteraccion": i})
                Data_Error = Data_Error.append(value, ignore_index=False)

    rezagos = Data_Error.groupby("Rezago").mean()[["error_Prediccion"]]
    rezagos["Rezago"] = rezagos.index
    rezagos = rezagos.sort_values(by=['error_Prediccion'], ascending=True)
    rezagos = rezagos.index.tolist()[0:4]
    return  rezagos

def pronosticosMuerte(DataModelMuertes, rezagos,fechamax):
    datelist = pd.date_range(fechamax+ timedelta(1), periods=10)

    indexday = pd.DataFrame({"index": range(1, 11)})

    for re1 in rezagos:
        x = DataModelMuertes[['21-40 Años', '41-60 Años', '61-80 Años', '< 20 Años', '> 80 Años']]
        y = DataModelMuertes[['Muertes']]

        rowdata = x.shape[0]

        Xtrain = x.iloc[0:rowdata - re1]
        Ytrain = y.iloc[0 + re1:rowdata]
        Xtest = x.iloc[rowdata - re1:rowdata]

        modelo = linear_model.LinearRegression()
        modelo = modelo.fit(Xtrain, Ytrain)
        y_pred = modelo.predict(Xtest)

        y_pred1 = [x[0] for x in y_pred]
        y_pred = pd.DataFrame({re1: y_pred1})
        indexday = indexday.merge(y_pred, left_index=True, right_index=True)

    indexday = indexday.loc[:, indexday.columns != "index"]
    pronosticofinal = indexday.mean(axis=1, skipna=True)
    desviacion = indexday.std(axis=1, skipna=True)
    li = pronosticofinal - desviacion
    li = li.to_frame()
    ls = pronosticofinal + desviacion
    ls = ls.to_frame()
    pronosticofinal = pronosticofinal.to_frame()
    pronosticofinal = pronosticofinal.merge(li, left_index=True, right_index=True)
    pronosticofinal = pronosticofinal.merge(ls, left_index=True, right_index=True)
    pronosticofinal.columns = ["Pronostico", "LI", "LS"]
    fechas = pd.DataFrame({"Fecha": datelist})
    pronosticofinal = fechas.merge(pronosticofinal, left_index=True, right_index=True)
    return pronosticofinal

def pronosticosDeadCity(Sintomas_history,Muertes_history,ciudad):

    DataModelMuertes = BaseModeloMuertes(Sintomas_history=Sintomas_history, Muertes_history=Muertes_history,
                                         ciudad=ciudad)
    rezagos = TrainMuertes(DataModelMuertes=DataModelMuertes, predicciones=10, iniciogenerador=50)
    Pronos = pronosticosMuerte(DataModelMuertes = DataModelMuertes, rezagos = rezagos)

    return Pronos
#Data_covid, Infectados_history, Recuperados_history, Muertes_history, Sintomas_history = inputInformation(url="www.datos.gov.co")


