from ModelRecuperados import  GeneracionPronsoticos_Recuperados
from StructureInformation import inputInformation



from jinja2 import Template
str = open('Template/index.html', 'r').read()
template = Template(str)
str = template.render(muertos= 2000)
open('index.html', 'w').write(str);

if __name__ == "__main__":
    Infectados_history, Recuperados_history, Muertes_history = inputInformation(url="www.datos.gov.co")
    GeneracionPronsoticos_Recuperados(Infectados_history = Infectados_history,Recuperados_history = Recuperados_history)

