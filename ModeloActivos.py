from ModelRecuperados import  PronosticosRecuperados
from ModeloMuertes import  pronosticosDeadCity
from ModeloInfectados import pronosticos_fecha
from StructureInformation import inputInformation
from DescriptiveAndPlots import grafica_series_activos
import pandas as pd

#Data_covid,Infectados_history, Recuperados_history,Muertes_history,Sintomas_history = inputInformation(url="www.datos.gov.co")

def Cumulative(lists):
    cu_list = []
    length = len(lists)
    cu_list = [sum(lists[0:x:1]) for x in range(0, length+1)]
    return cu_list[1:]

def activos(Infectados_history, Muertes_history, Recuperados_history, ciudad, PronosticosBog, pronosticosBog, p_b):
    Bogota_rec = Recuperados_history[Recuperados_history['ciudad_de_ubicaci_n'] == ciudad]
    list_Recuperados_history = Bogota_rec['id_de_caso'].tolist()
    list_Recuperados_history = list_Recuperados_history + PronosticosBog['Pronostico'].tolist()
    recu_bog = Cumulative(list_Recuperados_history)
    fecha_rec_bog = Bogota_rec['fecha_recuperado'].tolist() + PronosticosBog['Fecha'].tolist()
    acumulado_rec_bog = pd.DataFrame({'fecha': fecha_rec_bog, 'Recuperado': recu_bog})

    Bogota_inf = Infectados_history[Infectados_history['ciudad_de_ubicaci_n'] == ciudad]
    list_Infectados_history = Bogota_inf['id_de_caso'].tolist()
    list_Infectados_history = list_Infectados_history + p_b['Pronostico'].tolist()
    infe_bog = Cumulative(list_Infectados_history)
    fecha_inf_bog = Bogota_inf['fecha_diagnostico'].tolist() + p_b['Fecha'].tolist()
    acumulado_inf_bog = pd.DataFrame({'fecha': fecha_inf_bog, 'Infectados': infe_bog})

    Bogota_muer = Muertes_history[Muertes_history['ciudad_de_ubicaci_n'] == ciudad]
    list_Muertes_history = Bogota_muer['id_de_caso'].tolist()
    list_Muertes_history = list_Muertes_history + pronosticosBog['Pronostico'].tolist()
    muer_bog = Cumulative(list_Muertes_history)
    fecha_muer_bog = Bogota_muer['fecha_de_muerte'].tolist() + pronosticosBog['Fecha'].tolist()
    acumulado_muer_bog = pd.DataFrame({'fecha': fecha_muer_bog, 'Muertos': muer_bog})

    consolidado = acumulado_inf_bog.merge(acumulado_rec_bog, on=['fecha'], how='left')
    consolidado = consolidado.merge(acumulado_muer_bog, on=['fecha'], how='left')
    consolidado = consolidado.fillna(0)

    consolidado['Activos'] = consolidado['Infectados'] - consolidado['Recuperado'] - consolidado['Muertos']
    consolidado = consolidado[['fecha', 'Activos']]

    return consolidado

def GeneracionPronsoticos_Activos(Infectados_history, Muertes_history, Recuperados_history,Sintomas_history):

    PronosticosBog = PronosticosRecuperados(Infectados_history=Infectados_history,
                                            Recuperados_history=Recuperados_history,
                                            ciudad='Bogotá D.C.',
                                            numberLag=25,
                                            predicciones=10)

    pronosticosBog = pronosticosDeadCity(Sintomas_history=Sintomas_history,
                                         Muertes_history=Muertes_history,
                                         ciudad='Bogotá D.C.')

    p_b = pronosticos_fecha(Infectados_history=Infectados_history, ciudad="Bogotá D.C.")

    activos_bogota = activos(Infectados_history=Infectados_history, Muertes_history=Muertes_history,
                             Recuperados_history=Recuperados_history, ciudad='Bogotá D.C.',
                             PronosticosBog=PronosticosBog, pronosticosBog=pronosticosBog, p_b=p_b)

    grafica_series_activos(acumulado=activos_bogota, ciudad_name='Bogota', tipo_pronostico='Activos')

    Pronosticosmed = PronosticosRecuperados(Infectados_history=Infectados_history,
                                            Recuperados_history=Recuperados_history,
                                            ciudad='Medellín',
                                            numberLag=25,
                                            predicciones=10)

    pronosticosMed = pronosticosDeadCity(Sintomas_history=Sintomas_history,
                                         Muertes_history=Muertes_history,
                                         ciudad='Medellín')

    p_m = pronosticos_fecha(Infectados_history=Infectados_history, ciudad="Medellín")

    activos_medellin = activos(Infectados_history=Infectados_history, Muertes_history=Muertes_history,
                               Recuperados_history=Recuperados_history, ciudad='Medellín',
                               PronosticosBog=Pronosticosmed, pronosticosBog=pronosticosMed, p_b=p_m)

    grafica_series_activos(acumulado=activos_medellin, ciudad_name='Medellin', tipo_pronostico='Activos')

    Pronosticoscali = PronosticosRecuperados(Infectados_history=Infectados_history,
                                             Recuperados_history=Recuperados_history,
                                             ciudad='Cali',
                                             numberLag=25,
                                             predicciones=10)

    pronosticosCali = pronosticosDeadCity(Sintomas_history=Sintomas_history,
                                          Muertes_history=Muertes_history,
                                          ciudad='Cali')

    p_ca = pronosticos_fecha(Infectados_history=Infectados_history, ciudad="Cali")

    activos_cali = activos(Infectados_history=Infectados_history, Muertes_history=Muertes_history,
                           Recuperados_history=Recuperados_history, ciudad='Cali',
                           PronosticosBog=Pronosticoscali, pronosticosBog=pronosticosCali, p_b=p_ca)

    grafica_series_activos(acumulado=activos_cali, ciudad_name='Cali', tipo_pronostico='Activos')

    PronosticosBarra = PronosticosRecuperados(Infectados_history=Infectados_history,
                                              Recuperados_history=Recuperados_history,
                                              ciudad='Barranquilla',
                                              numberLag=25,
                                              predicciones=10)

    pronosticosBarra = pronosticosDeadCity(Sintomas_history=Sintomas_history,
                                           Muertes_history=Muertes_history,
                                           ciudad='Barranquilla')

    p_ba = pronosticos_fecha(Infectados_history=Infectados_history, ciudad="Barranquilla")

    activos_barra = activos(Infectados_history=Infectados_history, Muertes_history=Muertes_history,
                            Recuperados_history=Recuperados_history, ciudad='Barranquilla',
                            PronosticosBog=PronosticosBarra, pronosticosBog=pronosticosBarra, p_b=p_ba)

    grafica_series_activos(acumulado=activos_barra, ciudad_name='Barranquilla', tipo_pronostico='Activos')

    PronosticosCarta = PronosticosRecuperados(Infectados_history=Infectados_history,
                                              Recuperados_history=Recuperados_history,
                                              ciudad='Cartagena de Indias',
                                              numberLag=25,
                                              predicciones=10)

    pronosticosCarta = pronosticosDeadCity(Sintomas_history=Sintomas_history,
                                           Muertes_history=Muertes_history,
                                           ciudad='Cartagena de Indias')

    p_c = pronosticos_fecha(Infectados_history=Infectados_history, ciudad="Cartagena de Indias")

    activos_cartagena = activos(Infectados_history=Infectados_history, Muertes_history=Muertes_history,
                                Recuperados_history=Recuperados_history, ciudad='Cartagena de Indias',
                                PronosticosBog=PronosticosCarta, pronosticosBog=pronosticosCarta, p_b=p_c)

    grafica_series_activos(acumulado=activos_cartagena, ciudad_name='Cartagena', tipo_pronostico='Activos')
    print('Muy bien')
























