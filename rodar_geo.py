from functions import GeoEstimation, social_dataframe, tendencia_mensal2, tendencia_brasil, tendencia_mensal, get_municip_real
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.errors import ShapelyDeprecationWarning
import numpy as np
from autoaede_functions import plot_lisa, otimizar_k, weights_matrix, read_geodata, significant_HH, plot_lisa_BV, otimizar_k_BV, significant_HH_BV
import json
from urllib.request import urlopen
import geobr


app = ['Renner', 'Shein', 'Riachuelo', 'C&A', 'cea', 'Marisa']
pais = 'BR'

state_sigla = {'SP':'State of São Paulo',
'RJ': 'State of Rio de Janeiro',
'MG': 'State of Minas Gerais',
'ES': 'State of Espírito Santo',
'AC': 'State of Acre',
'AL': 'State of Alagoas',
'AP': 'State of Amapá',
'AM': 'State of Amazonas',
'PR': 'State of Paraná',
'BA': 'State of Bahia',
'CE': 'State of Ceará',
'DF': 'State of Distrito Federal',
'ES': 'State of Espírito Santo',
'GO': 'State of Goiás',
'MA': 'State of Maranhão',
'MT': 'State of Mato Grosso',
'MS': 'State of Mato Grosso do Sul',
'PA': 'State of Pará',
'PB': 'State of Paraíba',
'SE': 'State of Sergipe',
'TO': 'State of Tocantins',
'RN': 'State of Rio Grande do Norte',
'RS': 'State of Rio Grande do Sul',
'RO': 'State of Rondônia',
'RR': 'State of Roraima',
'SC': 'State of Santa Catarina',
'PE': 'State of Pernambuco',
'PI': 'State of Piauí',
}

state_all = list(state_sigla.keys())
state = [e for e in state_all if e not in ('Nenhum')]
#state = ['AP']


print(state)

data_inicial = '01-01-2022'
data_final = '26-09-2022'

lista_cores = ['Oranges', 'Greys', 'Blues', 'Greens', 'Purples', 'Reds'] # ['Oranges', 'Greys', 'Blues', 'Greens', 'Purples', 'Reds']

print(len(lista_cores))
print(len(app))
app_color_dict = pd.DataFrame({'app':app, 'cor':lista_cores})
adapt_cores = [i.lower()[:-1] for i in lista_cores]
##################################################

def start_read():
    dicionario_arquivos = {}
    geo = GeoEstimation(app, pais, start_date=data_inicial, final_date=data_final)
    try:
     state_df = social_dataframe(app, 'BR', estado=state, start_date=data_inicial, final_date=data_final, dicionario = dicionario_arquivos)
    except:
      state_df = None
    return geo, state_df, dicionario_arquivos
#state_df = []


def mapa_brasil():
    for row, value in app_color_dict.iterrows():
        GeoEstimation(value['app'], pais, start_date=data_inicial, final_date=data_final).map(cor=value['cor'], dicionario = dicionario_arquivos)

def mapa_estados():
     for row, value in app_color_dict.iterrows():
        lista_cores = ['Oranges', 'Greys', 'Blues', 'Greens', 'Purples', 'Reds']
        #app_color_dict = pd.DataFrame({'app':app, 'cor':lista_cores})
        print(value['app'])
        print(j)
        GeoEstimation(value['app'], 'BR', start_date=data_inicial, final_date=data_final).municip_map(j, cor = value['cor'], dicionario=dicionario_arquivos)

def clusters():
    dicio_cluster = {}
    set_list = []
    for k in app:
      dado = read_geodata(dicionario_arquivos[f'{k}_{j}_with_geometry'])
      #dado = state_df
      try:
          i_moran = otimizar_k(dado, k, 1, 10, p_value=0.05)
          pesos = weights_matrix(dado, metric = 'knn', k = i_moran)
          clusters = plot_lisa(dado, k, weights= pesos, k_opt=i_moran, estado=j)
          dicio_cluster.update({k:significant_HH(dado, k, weight= pesos)})
          set_list.append(set(dicio_cluster[k]))
      except:
        pass

def tendencia():
    app_color_dict['cor'] = adapt_cores
    retorno_df = tendencia_mensal2(app, j, adapt_cores, data_inicial = data_inicial, data_final = data_final)

    fig, ax = plt.subplots(1,1, figsize=(10, 6), dpi=200)
    retorno_df[app_color_dict['app']].plot(ax=ax,label=app_color_dict['app'], color=[i[:-1] if i[-1] == 's' else i for i in app_color_dict['cor']])

    plt.legend(app)
    plt.rcParams.update({"font.size": 10})
    plt.ylim(0,100)

    plt.title(f'Proporção mensal das buscas\n({retorno_df["geoName"][0]}, {data_inicial.split("-")[2]})', fontsize=15)
    plt.ylabel('Intensidade das buscas')
    plt.box(False)
    plt.savefig(f'maps/proporcao_{app}_{j}.png')


##################################################

geo = start_read()[0]
state_df = start_read()[1]
dicionario_arquivos = start_read()[2]

##################################################


# Mapa Brasil
mapa_brasil()

# Mapa dos Estados
for j in state:
    mapa_estados()

# Clusters
for j in state:
    clusters()

# Tendência
for j in state:
    tendencia()