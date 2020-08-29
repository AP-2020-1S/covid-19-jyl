import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets, linear_model, metrics
from sklearn.model_selection import train_test_split
import pandas as pd

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

Data_Modelo = ModelData(DataX = Infectados_history, DataY = Recuperados_history, ciudad = "Medellín",numberLag = 25)


Variables_Independientes = ['Rezago24','Rezago23','Rezago22','Rezago21']
Variable_Dependiente = ['Numero_de_Recuperados']


X = Data_Modelo[Variables_Independientes].as_matrix()
Y = infectadosrezagados[['Numero_de_Recuperados']]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,
                                                    random_state=1)


plt.scatter(X_train, y_train,  color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)