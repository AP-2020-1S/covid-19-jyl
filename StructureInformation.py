import pandas as pd
from sodapy import Socrata
import numpy as np

def get_information(url,limit):
    client = Socrata(url, None)
    data = client.get("gt2j-8ykr", limit= limit)
    data = pd.DataFrame.from_records(data)
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

    values = ['< 20 Años', '21-30 Años', '41-60 Años', '61-80 Años', '> 80 Años']

    data_frame_Covid_19['EdadCate'] = np.select(conditions, values)
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

Data_covid = get_information(url = "www.datos.gov.co",limit = 40000000)
Data_covid = filter_city_categorical_values(dataframe = Data_covid,
                                            filterCity = ['Bogotá D.C.',
                                                            'Medellín',
                                                            'Cali',
                                                            'Barranquilla',
                                                            'Cartagena de Indias'])

Infectados_history = structuredata(dataframe = Data_covid, variable = "fecha_diagnostico")
Recuperados_history = structuredata(dataframe = Data_covid, variable = "fecha_recuperado")
Muertes_history = structuredata(dataframe = Data_covid, variable = "fecha_de_muerte")

