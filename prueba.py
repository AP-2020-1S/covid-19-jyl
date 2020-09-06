from sodapy import Socrata
import numpy as np
from datetime import datetime
from datetime import date
import matplotlib.pylab as plt
%matplotlib inline
import statistics

import pandas as pd
import plotly.offline as pyo
import plotly.graph_objs as go

Infectados_history = pd.read_csv('bases/bd_infectados.csv')
Muertes_history = pd.read_csv('bases/Muertes_history.csv')
Recuperados_history = pd.read_csv('bases/Recuperados_history.csv')


def base_sir(Infectados_history, Muertes_history, Recuperados_history):
    sir_I = Infectados_history[Infectados_history['ciudad_de_ubicaci_n'] == 'Medellín']
    sir_I = sir_I[['fecha_diagnostico', 'id_de_caso']]
    sir_I['fecha_diagnostico'] = pd.to_datetime(sir_I['fecha_diagnostico'], format="%Y/%m/%d")
    sir_I = sir_I.set_index('fecha_diagnostico', )
    sir_I.rename(columns={"fecha_diagnostico": "fecha_diagnostico",
                          "id_de_caso": "infectados"},
                 inplace=True)
    sir_R = Recuperados_history[Recuperados_history['ciudad_de_ubicaci_n'] == 'Medellín']
    sir_R = sir_R[['fecha_recuperado', 'id_de_caso']]
    sir_R['fecha_recuperado'] = pd.to_datetime(sir_R['fecha_recuperado'], format="%Y/%m/%d")
    sir_R = sir_R.set_index('fecha_recuperado', )
    sir_R.rename(columns={"fecha_recuperado": "fecha_recuperado",
                          "id_de_caso": "recuperado"},
                 inplace=True)
    sir_M = Muertes_history[Muertes_history['ciudad_de_ubicaci_n'] == 'Medellín']
    sir_M = sir_M[['fecha_de_muerte', 'id_de_caso']]
    sir_M['fecha_de_muerte'] = pd.to_datetime(sir_M['fecha_de_muerte'], format="%Y/%m/%d")
    sir_M = sir_M.set_index('fecha_de_muerte', )
    sir_M.rename(columns={"fecha_de_muerte": "fecha_de_muerte",
                          "id_de_caso": "muerto"},
                 inplace=True)
    sir = pd.concat([sir_I, sir_R, sir_M], axis=1)
    sir = sir.fillna(0)
    return sir

sir= base_sir(Infectados_history=Infectados_history,
              Muertes_history=Muertes_history,
              Recuperados_history=Recuperados_history)

copia=sir
N = 2427000
copia['S0']= N -copia['infectados']- copia['recuperado']
copia['t_contagios']=0
copia['t_recuperados']=0
copia['t_muertos']=0
copia['Activos']=0
copia['Activos'][0]=1


base = copia
X_Value = base[["infectados"]]
base = pd.concat([base, X_Value.shift()], axis=1)
base.columns =['infectados', 'recuperado', 'muerto', 'S0', 't_contagios',
       't_recuperados', 't_muertos', 'Activos', 'infectadosrezago']

rezago = base[['infectados','infectadosrezago']]
rezago = rezago.dropna()
rezago["tasacontagio"] = (rezago['infectados'] / rezago['infectadosrezago'])

N = 2427000
tcontagio = 0.05
trecuperacion = 0.02
t_muerte = 0.005
so = base['S0'].tolist()
contagios = base['t_contagios'].tolist()
recuperados = base['t_recuperados'].tolist()
muertos = base['t_muertos'].tolist()
Activos = base['Activos'].tolist()
nrow = len(Activos)

for x in range(1,nrow):
    activos_value = Activos[x-1]
    so_value = so[x-1]
    tasa_c =(tcontagio * activos_value * so_value) / N
    contagios[x] = tasa_c
    tasa_r = trecuperacion * activos_value
    recuperados[x] = tasa_r
    tasa_m = t_muerte * activos_value
    muertos[x] = tasa_m
    Activos[x] = activos_value + tasa_c - tasa_r - tasa_m


simulacionsir= {'Activos': Activos, 'Muertos': muertos,'Recuperados': recuperados, 'Contagiados': contagios}
sirfinal= pd.DataFrame(data=simulacionsir)
sirfinal

###########################################################333

#### original code

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Total population, N.
N = 2427000
# Initial number of infected and recovered individuals, I0 and R0.
I0, R0 = 1, 0
# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - R0
# Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
beta, gamma = 0.5, 0.25
# A grid of time points (in days)
t = np.linspace(0, 180, 180)

# The SIR model differential equations.
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# Initial conditions vector
y0 = S0, I0, R0
# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, beta, gamma))
S, I, R = ret.T


# Plot the data on three separate curves for S(t), I(t) and R(t)
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=I,mode='lines',name='Infected'))
fig.add_trace(go.Scatter(x=t, y=S,mode='lines',name='suscep'))
fig.add_trace(go.Scatter(x=t, y=R,mode='lines',name='recuper'))
fig.show()