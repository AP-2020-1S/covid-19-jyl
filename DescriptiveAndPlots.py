import matplotlib.pyplot as plot
import seaborn as sns
import numpy as np

def barplotciudades(Data_covid, pathsave,title):
    ciudadesinfectados = Data_covid.groupby("ciudad_de_ubicaci_n").count()["id_de_caso"].sort_values(
        ascending=False).head(10)
    ciudadesinfectados = ciudadesinfectados.reset_index()
    ciudadesinfectados.columns = ["Ciudad", "Cantidad de infectados"]
    plot.figure()
    sns.set_style("darkgrid")
    bar_plot = sns.barplot(x=ciudadesinfectados["Ciudad"], y=ciudadesinfectados["Cantidad de infectados"],
                           palette="muted" ).set_title(title)


    bar_plot.get_figure().savefig( pathsave+  '.png')
    plot.figure()
    print("grafico ciudades guardado")

def pieplotsexo(Data_covid,pathsave,name):
    conditions = [
        (Data_covid['sexo'] == "F"),
        (Data_covid['sexo'] == "f"),
        (Data_covid['sexo'] == "M"),
        (Data_covid['sexo'] == "m"),
    ]

    values = ['Femenino', 'Femenino', 'Masculino', 'Masculino']

    Data_covid['sexo'] = np.select(conditions, values)

    sexo = Data_covid.groupby("sexo").count()["id_de_caso"].reset_index()


    pie = plot.pie(sexo["id_de_caso"], labels=sexo["sexo"])[0]
    fig = pie[0].get_figure()
    fig.savefig(pathsave + name + '.png')
    fig.clf()

def plot_edad(Data_covid,pathsave,name,xasis,yaxis,title):
    Data_covid['edad'] = Data_covid['edad'].astype(int)

    conditions = [
        (Data_covid['edad'] <= 20),
        (Data_covid['edad'] > 20) & (Data_covid['edad'] <= 40),
        (Data_covid['edad'] > 40) & (Data_covid['edad'] <= 60),
        (Data_covid['edad'] > 60) & (Data_covid['edad'] <= 80),
        (Data_covid['edad'] > 80)
    ]

    values = ['< 20 Años', '21-30 Años', '41-60 Años', '61-80 Años', '> 80 Años']

    Data_covid['EdadCate'] = np.select(conditions, values)

    edadinfectados = Data_covid.groupby("EdadCate").count()["id_de_caso"].sort_values(
        ascending=False).head(10)
    edadinfectados = edadinfectados.reset_index()
    edadinfectados.columns = ["Edad", "Cantidad de infectados"]
    plot.figure()
    sns.set_style("darkgrid")
    bar_plot = sns.barplot(x=edadinfectados["Edad"], y=edadinfectados["Cantidad de infectados"],
                           palette="muted").set_title(title)
    plot.xlabel(xasis)
    plot.ylabel(yaxis)

    bar_plot.get_figure().savefig(pathsave + name + '.png')
    plot.figure()
    print("grafico ciudades guardado")

def metricas(Data_covid):
    infectados = Data_covid.shape[0]
    muertos = len(Data_covid["fecha_de_muerte"].dropna())
    Recuperados = len(Data_covid["fecha_recuperado"].dropna())
    Activos = infectados - muertos - Recuperados

    datos = {"Infectados": infectados,
             "Muertos": muertos,
             "Recuperados": Recuperados,
             "Activos": Activos}

    return datos

def PlotsColombia(Data_covid):
    pieplotsexo(Data_covid=Data_covid, pathsave='fig/Colombia/', name="sexo")
    barplotciudades(Data_covid=Data_covid, pathsave='fig/Colombia/ciudades', title="Infectados vs ciudad")
    plot_edad(Data_covid=Data_covid, pathsave='fig/Colombia/', name="Edades",
              xasis="Edad", yaxis="Cantidad de Infectados", title="Infectados por Edad Colombia")
    valores = metricas(Data_covid=Data_covid)

    muertos = Data_covid[Data_covid["fecha_de_muerte"].notnull()]

    plot_edad(Data_covid=muertos, pathsave='fig/Colombia/',
              name="Muertos", xasis="Edad", yaxis="Cantidad de Muertes", title="Muertes por Edad Colombia")
    return valores

def plotsCiudad(Data_covid,pathsave,ciudad):

    Data_covid = Data_covid[Data_covid["ciudad_de_ubicaci_n"] == ciudad]
    pieplotsexo(Data_covid=Data_covid, pathsave=pathsave, name="sexo")
    plot_edad(Data_covid=Data_covid, pathsave=pathsave, name="Edades",
              xasis="Edad", yaxis="Cantidad de Infectados", title="Infectados por Edad " + ciudad)
    valores = metricas(Data_covid=Data_covid)

    muertos = Data_covid[Data_covid["fecha_de_muerte"].notnull()]

    plot_edad(Data_covid=muertos, pathsave=pathsave,
              name="Muertos", xasis="Edad", yaxis="Cantidad de Muertes", title="Muertes por Edad "+ ciudad)
    return valores






