from sklearn import  linear_model
import pandas as pd
import numpy as np
from datetime import timedelta
import re


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


    namescolumns = dataModel.columns.tolist() + ["Rezago" + str(x) for x in range(1, Rezagos.shape[1] - 2)]
    Rezagos.columns = namescolumns
    return Rezagos

def error_cuadratico(estimador, x,y_real):
    y_pred = estimador.predict(x)
    return np.sum((y_real - y_pred)**2/len(y_real))

def model_Rezagos(Data_Modelo,Variables_Independientes,Variable_Dependiente,predicciones):
    rowsdata = Data_Modelo.shape[0] - predicciones + 1
    inicio = rowsdata - 10

    Data_Error = pd.DataFrame(columns=["Rezago", "error_Prediccion", "iteraccion"])

    for i in range(inicio, rowsdata):
        train11 = Data_Modelo.iloc[0:i]
        test11 = Data_Modelo.iloc[i:i + predicciones]
        X_train = train11[Variables_Independientes]
        X_test = test11[Variables_Independientes]
        y_train = train11[Variable_Dependiente]
        y_test = test11[Variable_Dependiente]
        print(i)
        for x in Variables_Independientes:
            X_train1 = X_train[[x]]
            X_test1 = X_test[[x]]
            modelo = linear_model.LinearRegression()
            modelo = modelo.fit(X_train1, y_train)
            error = error_cuadratico(estimador=modelo, x=X_test1, y_real=y_test)
            value = pd.DataFrame({"Rezago": x,
                                  "error_Prediccion": error, "iteraccion": i})
            Data_Error = Data_Error.append(value, ignore_index=False)

        # Data_Error = Data_Error.sort_values(by=['error_Prediccion'], ascending=True)

    rezagos = Data_Error.groupby("iteraccion").min("error_Prediccion")
    rezagos = pd.merge(rezagos, Data_Error, on="error_Prediccion", how="left")
    rezagos = rezagos.groupby("Rezago").count()
    rezagos = rezagos.sort_values(by=['error_Prediccion'], ascending=False)
    rezagos =rezagos.index.tolist()[0:4]
    return rezagos

def Datatest(rezagos,Data_Modelo):
    datelist = pd.date_range(Data_Modelo["Fecha"].max() + timedelta(1), periods=10)

    Data_test = pd.DataFrame(columns=rezagos)

    for daylist in datelist:
        res = [int(re.findall(r'\d+', sub)[0]) for sub in rezagos]

        lis = []
        for x in res:
            lis.append(daylist - timedelta(x))

        dia = pd.DataFrame(data={'Fecha': lis})

        dia = pd.merge(dia, Data_Modelo, on="Fecha", how="left")[["Numero_de_Infectados"]]
        dia = dia.transpose()
        dia.columns = rezagos
        Data_test = Data_test.append(dia, ignore_index=False)
    return datelist ,Data_test

def funpronostico(Data_Modelo_Test, Data_Modelo_Train,rezagos,Variables_Independientes,Variable_Dependiente):
    indexday = pd.DataFrame({"index": range(1, 11)})

    for pronosticos in rezagos:
        Xtrain = Data_Modelo_Train[Variables_Independientes]
        Ytrain = Data_Modelo_Train[Variable_Dependiente]

        X_trainReza = Xtrain[[pronosticos]]
        X_testReza = Data_Modelo_Test[[pronosticos]]

        modelofinal = linear_model.LinearRegression()
        modelofinal.fit(X_trainReza, Ytrain)
        y_pred = modelofinal.predict(X_testReza)
        y_pred1 = [x[0] for x in y_pred]
        y_pred = pd.DataFrame({pronosticos: y_pred1})
        indexday = indexday.merge(y_pred, left_index=True, right_index=True)
    indexday = indexday.loc[:, indexday.columns != "index"]
    #indexday = indexday.iloc[:, [1, 2]]
    pronosticofinal = indexday.mean(axis=1, skipna=True)
    return pronosticofinal

def PronosticosRecuperados(Infectados_history,Recuperados_history,ciudad,numberLag,predicciones):
    Data_Modelo = ModelData(DataX=Infectados_history, DataY=Recuperados_history, ciudad=ciudad, numberLag=numberLag)
    Data_Modelo_Train = Data_Modelo.dropna()

    Variables_Independientes = ["Rezago" + str(x) for x in range(predicciones, numberLag)]
    Variable_Dependiente = ['Numero_de_Recuperados']
    selectrowws = ["Fecha", "Numero_de_Recuperados"] + ["Rezago" + str(x) for x in range(predicciones, numberLag)]
    Data_Modelo_Train = Data_Modelo_Train[selectrowws]

    Data_Modelo_Train = Data_Modelo_Train.sort_values(by=['Fecha'], ascending=True)

    rezagos = model_Rezagos(Data_Modelo=Data_Modelo_Train, Variables_Independientes=Variables_Independientes,
                            Variable_Dependiente=Variable_Dependiente, predicciones=predicciones)

    fechas, Data_Modelo_Test = Datatest(rezagos=rezagos, Data_Modelo=Data_Modelo)

    Pronostico = funpronostico(Data_Modelo_Test=Data_Modelo_Test,
                               Data_Modelo_Train = Data_Modelo_Train,
                               rezagos=rezagos,
                               Variables_Independientes=Variables_Independientes,
                               Variable_Dependiente=Variable_Dependiente)

    fechas = pd.DataFrame({"Fecha": fechas})
    Pronostico = Pronostico.to_frame()
    Pronostico.columns = ["Pronostico"]
    Pronostico = fechas.merge(Pronostico, left_index=True, right_index=True)
    return Pronostico

def GeneracionPronsoticos_Recuperados(Infectados_history,Recuperados_history):
    PronosticosBog = PronosticosRecuperados(Infectados_history=Infectados_history,
                                            Recuperados_history=Recuperados_history,
                                            ciudad='Bogotá D.C.',
                                            numberLag=25,
                                            predicciones=10)

    Pronosticosmed = PronosticosRecuperados(Infectados_history=Infectados_history,
                                            Recuperados_history=Recuperados_history,
                                            ciudad='Medellín',
                                            numberLag=25,
                                            predicciones=10)

    Pronosticoscali = PronosticosRecuperados(Infectados_history=Infectados_history,
                                             Recuperados_history=Recuperados_history,
                                             ciudad='Cali',
                                             numberLag=25,
                                             predicciones=10)

    PronosticosBarra = PronosticosRecuperados(Infectados_history=Infectados_history,
                                              Recuperados_history=Recuperados_history,
                                              ciudad='Barranquilla',
                                              numberLag=25,
                                              predicciones=10)

    PronosticosCarta = PronosticosRecuperados(Infectados_history=Infectados_history,
                                              Recuperados_history=Recuperados_history,
                                              ciudad='Cartagena de Indias',
                                              numberLag=25,
                                              predicciones=10)

    return PronosticosBog,Pronosticosmed,Pronosticoscali,PronosticosBarra,PronosticosCarta







