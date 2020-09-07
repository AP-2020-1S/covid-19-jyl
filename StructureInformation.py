import pandas as pd
from sodapy import Socrata
import numpy as np


def get_information(url,limit):
    client = Socrata(url, None)
    data = client.get("gt2j-8ykr", limit= limit)
    data = pd.DataFrame.from_records(data)
    return data

def castDate(data,value):
    data[value] = data[value].str[:10]
    data[value] = pd.to_datetime(data[value])
    return data


def filter_city_categorical_values(dataframe, filterCity):
    data_frame_Covid_19 = dataframe[dataframe.ciudad_de_ubicaci_n.isin(filterCity)]
    data_frame_Covid_19['edad'] = data_frame_Covid_19['edad'].astype(int)

    conditions = [
       (data_frame_Covid_19['edad'] <= 20),
        (data_frame_Covid_19['edad'] > 20) & (data_frame_Covid_19['edad'] <= 40),
        (data_frame_Covid_19['edad'] > 40) & (data_frame_Covid_19['edad'] <= 60),
        (data_frame_Covid_19['edad'] > 60) & (data_frame_Covid_19['edad'] <= 80),
        (data_frame_Covid_19['edad'] > 80)
    ]

    values = ['< 20 Años', '21-40 Años', '41-60 Años', '61-80 Años', '> 80 Años']

    data_frame_Covid_19['EdadCate'] = np.select(conditions, values)
    data_frame_Covid_19 =castDate(data = data_frame_Covid_19,value = "fecha_de_notificaci_n")
    data_frame_Covid_19 = castDate(data=data_frame_Covid_19, value="fecha_diagnostico")
    data_frame_Covid_19 = castDate(data=data_frame_Covid_19, value="fecha_recuperado")
    data_frame_Covid_19 = castDate(data=data_frame_Covid_19, value="fecha_reporte_web")
    data_frame_Covid_19 = castDate(data=data_frame_Covid_19, value="fecha_de_muerte")


    return data_frame_Covid_19

def structuredata(dataframe,variable):

    dataframe = dataframe[dataframe[variable].notnull()]

    ## numero_casos
    nummbercase = dataframe.groupby([variable, "ciudad_de_ubicaci_n"]).count()["id_de_caso"].reset_index()

    ## numero_casos_edad
    numbercaseedada = dataframe.groupby([variable, "ciudad_de_ubicaci_n", "EdadCate"]).count()["id_de_caso"].reset_index()
    numbercaseedada = numbercaseedada.pivot_table(index=[variable, "ciudad_de_ubicaci_n"], columns="EdadCate", values="id_de_caso", aggfunc=np.sum,
                                fill_value=0).reset_index()

    result = pd.merge(nummbercase, numbercaseedada, on=[variable, 'ciudad_de_ubicaci_n'], how="left")
    return result

def structuredata_Sintomas(Data_covid1):
    fismuerte = Data_covid1[Data_covid1["fecha_de_muerte"].notnull()]
    fismuerte = fismuerte[fismuerte["fis"] != 'Asintomático']
    fismuerte['fis'] = fismuerte['fis'].str[:10]
    fismuerte['fis'] = pd.to_datetime(fismuerte['fis'])

    nummbercasefis = fismuerte.groupby(['fis', "ciudad_de_ubicaci_n"]).count()["id_de_caso"].reset_index()
    ## numero_casos_edad
    numbercaseedada = fismuerte.groupby(['fis', "ciudad_de_ubicaci_n", "EdadCate"]).count()["id_de_caso"].reset_index()
    numbercaseedada = numbercaseedada.pivot_table(index=['fis', "ciudad_de_ubicaci_n"], columns="EdadCate",
                                                  values="id_de_caso", aggfunc=np.sum,
                                                  fill_value=0).reset_index()

    result1 = pd.merge(nummbercasefis, numbercaseedada, on=['fis', 'ciudad_de_ubicaci_n'], how="left")

    fisGeneral = Data_covid1[Data_covid1["fis"] != 'Asintomático']
    fisGeneral['fis'] = fisGeneral['fis'].str[:10]
    fisGeneral['fis'] = pd.to_datetime(fisGeneral['fis'])

    nummbercasefisgene = fisGeneral.groupby(['fis', "ciudad_de_ubicaci_n"]).count()["id_de_caso"].reset_index()
    ## numero_casos_edad
    nummbercasedadfisgene = fisGeneral.groupby(['fis', "ciudad_de_ubicaci_n", "EdadCate"]).count()["id_de_caso"].reset_index()
    nummbercasedadfisgene = nummbercasedadfisgene.pivot_table(index=['fis', "ciudad_de_ubicaci_n"], columns="EdadCate",
                                                  values="id_de_caso", aggfunc=np.sum,
                                                  fill_value=0).reset_index()
    result2 = pd.merge(nummbercasefisgene, nummbercasedadfisgene, on=['fis', 'ciudad_de_ubicaci_n'], how="left")
    result2.columns = ['fis', 'ciudad_de_ubicaci_n', 'id_de_caso General', '21-30 Años General', '41-60 Años General',
       '61-80 Años General', '< 20 Años General', '> 80 Años General']

    Resultadofinal = pd.merge(result1, result2, on=['fis', 'ciudad_de_ubicaci_n'], how="left")

    return  Resultadofinal


def inputInformation(url):
    print("Estamos obteniendo la informacion")
    Data_covid = get_information(url=url, limit=40000000)
    conditions = [
        (Data_covid['sexo'] == "F"),
        (Data_covid['sexo'] == "f"),
        (Data_covid['sexo'] == "M"),
        (Data_covid['sexo'] == "m"),
    ]

    values = ['Femenino', 'Femenino', 'Masculino', 'Masculino']

    Data_covid['sexo'] = np.select(conditions, values)

    Data_covid1 = filter_city_categorical_values(dataframe=Data_covid,
                                                filterCity=['Bogotá D.C.',
                                                            'Medellín',
                                                            'Cali',
                                                            'Barranquilla',
                                                            'Cartagena de Indias'])

    Infectados_history = structuredata(dataframe=Data_covid1, variable="fecha_diagnostico")
    Recuperados_history = structuredata(dataframe=Data_covid1, variable="fecha_recuperado")
    Muertes_history = structuredata(dataframe=Data_covid1, variable="fecha_de_muerte")
    Sintomas_history = structuredata_Sintomas(Data_covid1=Data_covid1)
    return Data_covid,Infectados_history, Recuperados_history,Muertes_history,Sintomas_history




