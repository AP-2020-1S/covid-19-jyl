
import matplotlib.pyplot as plot
import pandas as pd
from StructureInformation import inputInformation
import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt



Infectados_history, Recuperados_history, Muertes_history = inputInformation(url="www.datos.gov.co")



d = {'y': [1, 2,3,4,5], 'li': [0,1,2,3, 4],'ls': [2,3,4,5,6], 't': [3,4,5,6,7,]}
base = pd.DataFrame(data=d)

def grafico(base):
    base.rename(columns={'y': 'y'}, inplace=True)
    base.rename(columns={'li': 'LI'}, inplace=True)
    base.rename(columns={'ls': 'LS'}, inplace=True)
    base.rename(columns={'t': 't'}, inplace=True)
    base['y'] = [int(x) for x in base['y']]
    base['LI'] = [int(x) for x in base['LI']]
    base['LS'] = [int(x) for x in base['LS']]
    plot.plot('t', 'LI', data=base, marker='', color='blue', linewidth=3, linestyle=":")
    plot.plot('t', 'LS', data=base, marker='', color='blue', linewidth=3, linestyle=":")
    plot.plot('t', 'y', data=base, marker='', color='grey', linewidth=3, linestyle="-")
    plot.legend()

grafico(base = base)


import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


data = sm.datasets.co2.load_pandas()
y = data.data

y = y['co2'].resample('MS').mean()

# The term bfill means that we use the value before filling in missing values
y = y.fillna(y.bfill())

print(y)