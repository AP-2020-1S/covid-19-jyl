from StructureInformation import inputInformation
from DescriptiveAndPlots import grafica_series_infectados
import pandas as pd
from sodapy import Socrata
import numpy as np
from datetime import datetime,timedelta
from datetime import date
import matplotlib.pylab as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten
from statistics import mean
from sklearn import svm
import statistics as stats

from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import pandas as pd
import plotly.offline as pyo
import plotly.graph_objs as go

#Data_covid, Infectados_history, Recuperados_history, Muertes_history, Sintomas_history = inputInformation(url="www.datos.gov.co")

def datatrain(Infectados_history,ciudad):
    df = Infectados_history[Infectados_history['ciudad_de_ubicaci_n']==ciudad]
    df = df[['fecha_diagnostico', 'id_de_caso']]
    df['fecha_diagnostico'] = pd.to_datetime(df['fecha_diagnostico'], format="%Y/%m/%d")
    #df=df.drop(df.index[len(df)-1])
    df = df.set_index('fecha_diagnostico',)
    #Rezagos
    df["COVID t-1"] = df.loc[:,"id_de_caso"].shift(periods=1)
    df["COVID t-2"] = df.loc[:,"id_de_caso"].shift(periods=2)
    df["COVID t-3"] = df.loc[:,"id_de_caso"].shift(periods=3)
    df = df.drop([df.index[0], df.index[1], df.index[2] ], axis=0)
    #Variables Intrés
    df['week'] = df.index.week
    df['mes'] = df.index.month
    df['week_day'] = df.index.weekday
    #Partición x e y
    y = df[['id_de_caso']].to_numpy()
    X = df.drop(columns = ['id_de_caso','week','mes','week_day'])
    #train, test
    X_train = X[:int(X.shape[0]*0.85)]
    X_test = X[int(X.shape[0]*0.85):]
    y_train = y[:int(X.shape[0]*0.85)]
    y_test = y[int(X.shape[0]*0.85):]
    #Estandarizar
    standarscaler = StandardScaler()
    X_trained_scaled = standarscaler.fit_transform(X_train)
    X_trained_scaled = pd.DataFrame(X_trained_scaled, index=X_train.index, columns=X_train.columns.values)
    X_test_scaled = standarscaler.transform(X_test)
    X_test_scaled = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns.values)
    return X_train,X_test,y_train,y_test

def train_modelo_redes_neuronales(X_trained_scaled,X_test_scaled,y_train,y_test):
    trainX = X_trained_scaled
    testX = X_test_scaled
    trainy = y_train
    testy = y_test
    #Score
    clf = MLPRegressor(hidden_layer_sizes=(100, 20),activation='relu', max_iter=1000, random_state=42)
    clf.fit(trainX, trainy.ravel())
    accuracy=clf.score(testX, testy)
    return accuracy

def train_regresion_lineal(X_trained_scaled, X_test_scaled, y_train, y_test):
    trainX = X_trained_scaled
    testX = X_test_scaled
    trainy = y_train
    testy = y_test

    reg = LinearRegression().fit(trainX, trainy)
    accuracy = reg.score(testX, testy)
    return accuracy

def train_random_forest(X_trained_scaled, X_test_scaled, y_train, y_test):
    trainX = X_trained_scaled.to_numpy()
    testX = X_test_scaled.to_numpy()
    trainy = y_train
    testy = y_test

    regr = RandomForestRegressor(max_depth=2, random_state=0)
    regr.fit(trainX, trainy)
    accuracy = regr.score(testX, testy)
    return accuracy

def entrenamiento(X_trained_scaled, X_test_scaled, y_train, y_test):
    A=train_modelo_redes_neuronales(X_trained_scaled=X_trained_scaled,X_test_scaled=X_test_scaled,y_train=y_train,y_test=y_test)
    B=train_regresion_lineal(X_trained_scaled=X_trained_scaled,X_test_scaled=X_test_scaled,y_train=y_train,y_test=y_test)
    C=train_random_forest(X_trained_scaled=X_trained_scaled,X_test_scaled=X_test_scaled,y_train=y_train,y_test=y_test)
    Score=[A,B,C]
    return Score

def test_modelo_redes_neuronales(X_trained_scaled,X_test_scaled,y_train):
    trainX = X_trained_scaled
    testX = X_test_scaled
    trainy = y_train
    #Score
    clf = MLPRegressor(hidden_layer_sizes=(100, 20),activation='relu', max_iter=1000, random_state=42)
    clf.fit(trainX, trainy.ravel())
    y_pred = clf.predict(testX)
    return y_pred

def test_regresion_lineal(X_trained_scaled, X_test_scaled, y_train):
    trainX = X_trained_scaled
    testX = X_test_scaled
    trainy = y_train

    reg = LinearRegression().fit(trainX, trainy)
    y_pred = reg.predict(testX)
    return y_pred

def test_random_forest(X_trained_scaled, X_test_scaled, y_train):
    trainX = X_trained_scaled.to_numpy()
    testX = X_test_scaled.to_numpy()
    trainy = y_train

    regr = RandomForestRegressor(max_depth=2, random_state=0)
    regr.fit(trainX, trainy)
    y_pred = regr.predict(testX)
    return y_pred

def datatest(Infectados_history,ciudad):
    df = Infectados_history[Infectados_history['ciudad_de_ubicaci_n']==ciudad]
    df = df[['fecha_diagnostico', 'id_de_caso']]
    df['fecha_diagnostico'] = pd.to_datetime(df['fecha_diagnostico'], format="%Y/%m/%d")
    df=df.drop(df.index[len(df)-1])
    df = df.set_index('fecha_diagnostico',)
    #Rezagos
    df["COVID t-1"] = df.loc[:,"id_de_caso"].shift(periods=1)
    df["COVID t-2"] = df.loc[:,"id_de_caso"].shift(periods=2)
    df["COVID t-3"] = df.loc[:,"id_de_caso"].shift(periods=3)
    df = df.drop([df.index[0], df.index[1], df.index[2] ], axis=0)
    #Variables Intrés
    df['week'] = df.index.week
    df['mes'] = df.index.month
    df['week_day'] = df.index.weekday
    #Partición x e y
    y_train = df[['id_de_caso']].to_numpy()
    X_train = df.drop(columns = ['id_de_caso','week','mes','week_day'])
    t_3_test=df[["id_de_caso"]].tail(3)
    trasn=t_3_test.transpose()
    trasn.columns=["COVID t-3","COVID t-2","COVID t-1"]
    X_test=trasn[["COVID t-1","COVID t-2","COVID t-3"]]
    #Estandarizar
    standarscaler = StandardScaler()
    X_trained_scaled = standarscaler.fit_transform(X_train)
    X_trained_scaled = pd.DataFrame(X_trained_scaled, index=X_train.index, columns=X_train.columns.values)
    X_test_scaled = standarscaler.transform(X_test)
    X_test_scaled = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns.values)
    return X_train,X_test,y_train

def pronostico(X_trained_scaled,X_test_scaled,y_train):
    pronostico_redes=test_modelo_redes_neuronales(X_trained_scaled=X_trained_scaled,X_test_scaled=X_test_scaled,y_train=y_train)
    pronostico_regresion=test_regresion_lineal(X_trained_scaled=X_trained_scaled,X_test_scaled=X_test_scaled,y_train=y_train)
    pronostico_bosque=test_random_forest(X_trained_scaled=X_trained_scaled,X_test_scaled=X_test_scaled,y_train=y_train)
    pronostico_final=[pronostico_redes,pronostico_regresion,pronostico_bosque]
    a=np.array([pronostico_final])
    return [a.mean()-a.std(),a.mean(),a.mean()+a.std()]

#ciudad="Medellín"
#Infectados_history = Infectados_history[Infectados_history['ciudad_de_ubicaci_n']==ciudad]
#X_trained_scaled,X_test_scaled,y_train,y_test=datatrain(Infectados_history=Infectados_history,ciudad=ciudad)
#entrenamiento=entrenamiento(X_trained_scaled=X_trained_scaled,X_test_scaled=X_test_scaled,y_train=y_train,y_test=y_test)
#X_trained_scaled,X_test_scaled,y_train=datatest(Infectados_history=Infectados_history,ciudad=ciudad)

def fun_pronosticos(Infectados_history,ciudad):
    X_trained_scaled, X_test_scaled, y_train = datatest(Infectados_history=Infectados_history, ciudad=ciudad)
    p=pronostico(X_trained_scaled=X_trained_scaled,X_test_scaled=X_test_scaled,y_train=y_train)
    pronostico_F=pd.DataFrame({'LI':[p[0]],'Pronostico':[p[1]],'LS':[p[2]]})

    historicos=Infectados_history['id_de_caso'].tolist()
    i=1
    while i <=9:
        historicos = historicos + [p[1]]
        lon = len(historicos)
        xtest = {'COVID t-1': [historicos[lon - 1]], 'COVID t-2': [historicos[lon - 2]], 'COVID t-3': [historicos[lon - 3]]}
        xtest = pd.DataFrame(xtest)
        p = pronostico(X_trained_scaled=X_trained_scaled, X_test_scaled=xtest, y_train=y_train)
        p1=pd.DataFrame({'LI':[p[0]],'Pronostico':[p[1]],'LS':[p[2]]})
        pronostico_F=pronostico_F.append(p1,ignore_index=False)
        i+=1
    return (pronostico_F)

def pronosticos_fecha(Infectados_history,ciudad):
    fechas = pd.date_range(Infectados_history['fecha_diagnostico'].max() + timedelta(1), periods=10)
    fechas = fechas.to_frame()
    fechas = fechas.reset_index()[[0]]
    p_f = fun_pronosticos(Infectados_history=Infectados_history, ciudad=ciudad)
    p_f = p_f.reset_index()[['LI', 'Pronostico', 'LS']]
    p_f = fechas.merge(p_f, left_index=True, right_index=True)
    p_f.columns = ['Fecha', 'LI', 'Pronostico', 'LS']
    p_f = p_f[['Fecha', 'Pronostico', 'LI', 'LS']]
    return p_f

def GeneracionPronsoticos_infectados(Infectados_history):
    print("Estamos pronosticando")
    p_b=pronosticos_fecha(Infectados_history=Infectados_history,ciudad="Bogotá D.C.")

    grafica_series_infectados(Infectados_history=Infectados_history, pronostico=p_b, ciudad='Bogotá D.C.',
                          ciudad_name="Bogota",
                          tipo_pronostico="Infectados")

    p_m=pronosticos_fecha(Infectados_history=Infectados_history, ciudad="Medellín")

    grafica_series_infectados(Infectados_history=Infectados_history, pronostico=p_m, ciudad='Medellín',
                              ciudad_name="Medellin",
                              tipo_pronostico="Infectados")

    p_c=pronosticos_fecha(Infectados_history=Infectados_history, ciudad="Cartagena de Indias")

    grafica_series_infectados(Infectados_history=Infectados_history, pronostico=p_c, ciudad='Cartagena de Indias',
                              ciudad_name="Cartagena",
                              tipo_pronostico="Infectados")


    p_ca=pronosticos_fecha(Infectados_history=Infectados_history, ciudad="Cali")

    grafica_series_infectados(Infectados_history=Infectados_history, pronostico=p_ca, ciudad='Cali',
                              ciudad_name="Cali",
                              tipo_pronostico="Infectados")


    p_ba=pronosticos_fecha(Infectados_history=Infectados_history, ciudad="Barranquilla")

    grafica_series_infectados(Infectados_history=Infectados_history, pronostico=p_ba, ciudad='Barranquilla',
                              ciudad_name="Barranquilla",
                              tipo_pronostico="Infectados")
    print("Pronosticos realizados")
    return p_b,p_m,p_c,p_ca,p_ba

#ga=GeneracionPronsoticos_infectados(Infectados_history=Infectados_history)