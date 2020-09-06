from ModelRecuperados import  GeneracionPronsoticos_Recuperados
from StructureInformation import inputInformation
from DescriptiveAndPlots import plotsCiudad
from DescriptiveAndPlots import PlotsColombia


if __name__ == "__main__":

    Data_covid,Infectados_history, Recuperados_history,Muertes_history,Sintomas_history = inputInformation(url="www.datos.gov.co")
    GeneracionPronsoticos_Recuperados(Infectados_history=Infectados_history, Recuperados_history=Recuperados_history)

    colombia = PlotsColombia(Data_covid=Data_covid)
    Medellin = plotsCiudad(Data_covid=Data_covid, pathsave='fig/Medellin/', ciudad="Medellín")
    Cali = plotsCiudad(Data_covid=Data_covid, pathsave='fig/Cali/', ciudad="Cali")
    bogota = plotsCiudad(Data_covid=Data_covid, pathsave='fig/Bogota/', ciudad='Bogotá D.C.')
    Barranquilla = plotsCiudad(Data_covid=Data_covid, pathsave='fig/Barranquilla/', ciudad="Barranquilla")
    Cartagena = plotsCiudad(Data_covid=Data_covid, pathsave='fig/Cartagena/', ciudad="Cartagena de Indias")

    from jinja2 import Template

    str = open('Template/index.html', 'r').read()
    template = Template(str)
    str = template.render(InfectadosColombia=colombia.get("Infectados"),
                          RecuperadosColombia=colombia.get("Recuperados"),
                          ActivosColombia=colombia.get("Activos"),
                          MuertosColombia=colombia.get("Muertos"),
                          InfectadosBogota=bogota.get("Infectados"),
                          RecuperadosBogota=bogota.get("Recuperados"),
                          ActivosBogota=bogota.get("Activos"),
                          MuertosBogota=bogota.get("Muertos"),
                          InfectadosMedellin=Medellin.get("Infectados"),
                          RecuperadosMedellin=Medellin.get("Recuperados"),
                          ActivosMedellin=Medellin.get("Activos"),
                          MuertosMedellin=Medellin.get("Muertos"),
                          InfectadosCali=Cali.get("Infectados"),
                          RecuperadosCali=Cali.get("Recuperados"),
                          ActivosCali=Cali.get("Activos"),
                          MuertosCali=Cali.get("Muertos"),
                          InfectadosBarranquilla=Barranquilla.get("Infectados"),
                          RecuperadosBarranquilla=Barranquilla.get("Recuperados"),
                          ActivosBarranquilla=Barranquilla.get("Activos"),
                          MuertosBarranquilla=Barranquilla.get("Muertos"),
                          InfectadosCartagena=Cartagena.get("Infectados"),
                          RecuperadosCartagena=Cartagena.get("Recuperados"),
                          ActivosCartagena=Cartagena.get("Activos"),
                          MuertosCartagena=Cartagena.get("Muertos")

                          )
    open('index.html', 'w').write(str);

