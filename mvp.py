import pandas as pd
import numpy as np
import geobr
import matplotlib.pyplot as plt
from unidecode import unidecode
import googletrends as googletrends
from shapely import wkt
import warnings
from shapely.errors import ShapelyDeprecationWarning
#from etl_functions import find_mongo
from datetime import datetime
#from pymongo import MongoClient
import requests
from bs4 import BeautifulSoup
from pytrends.request import TrendReq
from pytrendAddons import interest_by_city
import geopandas as gpd
import re
import os
from dateutil.relativedelta import relativedelta
from dateutil.rrule import rrule, MONTHLY
import time
import streamlit as st
from autoaede_functions import plot_lisa, otimizar_k, weights_matrix, read_geodata, significant_HH
import networkx as nx
from sklearn import preprocessing
import plotly.express as px
import json
from urllib.request import urlopen

warnings.filterwarnings("ignore")





class GeoEstimation():
  def __init__(self, app, country, start_date, final_date):
    self.app = app
    self.country = country
    self.start_date = start_date
    self.final_date = final_date
    
  def dataframe(self, dicionario):
    '''
    Busca os dados espaciais do google Trends usando a lib googletrends
    '''
    df = googletrends.spatio(self.app, geo=[self.country], date_start=self.start_date,date_stop=self.final_date, method='')[self.country]['df'].reset_index()
    #pytrends = TrendReq(hl='pt-BR')
    #df = pytrends.build_payload(self.app, timeframe=f'{datetime.strptime(self.start_date, "%Y-%m-%d")} {datetime.strptime(self.final_date, "%Y-%m-%d")}', geo='BR')
    df['name_state'] = df['geoName'].apply(lambda x: x.split('State of ')[1].title() if 'State' in x else x)
    df['name_state'] = df['name_state'].apply(lambda x: 'Distrito Federal' if x == 'Federal District' else x)
    df_estados = geobr.read_state(year=2020)
    df_estados['name_state'] = df_estados['name_state'].apply(lambda x: unidecode(x))
    df['name_state'] = df['name_state'].apply(lambda x: unidecode(x))
    df_merged = df_estados.merge(df.drop(columns='geoName'), how='left', on='name_state').fillna(0)
    #df_merged[f'{self.app}_taxa'] = df_merged[self.app] / df_merged[self.app].sum()
    #df_merged['geo_downloads_estimation'] = abs(round(df_merged[f'{self.app}_taxa'] * GeoEstimation(self.app, self.country, self.start_date, self.final_date).search_appid()['var_downloads'],0))
    df_merged['geometry'] = df_merged['geometry'].astype('str').apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df_merged, crs='epsg:4326')
    dicionario.update({f'{self.app}_{self.country}_with_geometry' : gdf})
    #gdf.to_csv(f'excel_results/{self.country}/{self.app}_{self.country}_with_geometry.csv')
    return gdf
    
  def map(self, dicionario, cor): # novo parametro -> geoestimation_dataframe
    '''
    Plota o mapa dos estados pelo Índice Google Trends
    '''
    dado = GeoEstimation(self.app, self.country, self.start_date, self.final_date).dataframe(dicionario)
    #dado = geoestimation_dataframe
    
    plt.rcParams.update({"font.size": 10})
    fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
    fig = dado.plot(
        column=self.app,
        cmap=cor,
        legend=True,
        edgecolor='black',
        linewidth=0.3,
        legend_kwds={
            "label": "Intensidade das buscas",
            "orientation": "horizontal",
            "shrink": 0.6,
        },
        ax=ax,
    )
    ax.set_title(f'Pesquisas por "{self.app}"\nBrasil, ({self.start_date} ~ {self.final_date})', fontsize=15)
    ax.axis("off")
    #plt.savefig(f'maps/{self.app}_{self.country}_map.png')
    

  def get_municip(self, estado, dicionario):
        '''
        Essa função retorna os dados municipais dos Índices Google trends
        
        * Parâmetros:
        - estado: estado
        
        '''
        self.estado = estado
        inicio = datetime.strptime(self.start_date, '%d-%m-%Y').strftime('%Y-%m-%d')
        final = datetime.strptime(self.final_date, '%d-%m-%Y').strftime('%Y-%m-%d')
        #df_brasil = GeoEstimation(self.app, self.country, start_date=self.start_date, final_date=self.final_date).dataframe()
        pytrends = TrendReq(hl='pt-BR')
        pytrends.build_payload([self.app], timeframe=f'{inicio} {final}', geo=f'BR-{self.estado}')
        df_muni = interest_by_city(pytrends, inc_low_vol=True)
        #df_muni = df_muni[df_muni[self.app] >=1]
        #df_muni[f'{self.app}_taxa'] = df_muni[self.app] / df_muni[self.app].sum()
        dado = df_muni
        
        df_muni = df_muni.reset_index()
        
        #df_estado = df_brasil[df_brasil['abbrev_state'] == self.estado].reset_index()
        #print(df_brasil[df_brasil['abbrev_state'] == self.estado]['geo_rating_estimation'].values[0])
        
        df_final = pd.DataFrame()
        df_final['name_muni'] = df_muni['geoName'].str.title()
        df_final[self.app] = df_muni[self.app]
        dado_estado = geobr.read_municipality(code_muni='all', year=2020).query(f'abbrev_state == "{self.estado}"')
        #dado_estado = dado_estado[dado_estado['abbrev_state'] == self.estado]
        print(dado_estado)
        dado = dado_estado.merge(df_final, how='left', on='name_muni').fillna(0)
        #df_final['geo_downloads_estimation'] = abs(round(df_muni[f'{self.app}_taxa'] * df_brasil[df_brasil['abbrev_state'] == self.estado]['geo_downloads_estimation'].values[0]))
        #dado.to_csv(f'excel_results/{self.country}/{self.app}_{self.estado}_with_geometry.csv')
        dicionario.update({f'{self.app}_{self.estado}_with_geometry' : dado})
        return dado
    
  def municip_map(self, estado, cor, dicionario, geojson): #novo parametro -> geoestimation_get_municip
        '''
        Essa função retorna o mapa do município e do Índice Google Trends.
        
        * Parâmetros:
        - estado: estado
        - cor: cor
        '''
        self.estado = estado
        #dado_estado = geobr.read_municipality(code_muni=self.estado, year=2020)
        dado_estim= GeoEstimation(self.app, self.country, self.start_date, self.final_date).get_municip(self.estado, dicionario)
        #dado_estim = geoestimation_get_municip
        dado = gpd.GeoDataFrame(dado_estim, crs='epsg:4326')
        #dado = dado_estado.merge(dado_estim, how='left', on='name_muni').fillna(0)
        
        plt.rcParams.update({"font.size": 10})
        fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
        fig = dado.plot(
            column=self.app,
            cmap=cor,
            legend=True,
            edgecolor='black',
            linewidth=0.3,
            legend_kwds={
                "label": "Intensidade das buscas",
                "orientation": "horizontal",
                "shrink": 0.6,
            },
            ax=ax,
        )
        plt.title(f"Pesquisas por {self.app}\n{estado}, {self.start_date} ~ {self.final_date}", fontsize=15)
        #plt.title(f'{estado}, {self.start_date} ~ {self.final_date}', fontsize=8)
        ax.axis("off")
        '''
        fig = px.choropleth(
        #geo_final, #soybean database
        locations = dado['name_muni'], #define the limits on the map/geography
        geojson = geojson_sp, #shape information
        featureidkey='properties.name',
        color_continuous_scale=cor,
        color = dado[self.app], #defining the color of the scale through the database
        title = self.app, #title of the map
         #animation_frame = 'ano' #creating the application based on the year
        )
        fig.update_geos(fitbounds = "locations", visible = False)
        return fig
        #plt.savefig(f'maps/{self.app}_map_{self.estado}.png')'''
        
        
        
  def similar_keywords(self, estado):
        self.estado = estado
        inicio = datetime.strptime(self.start_date, '%d-%m-%Y').strftime('%Y-%m-%d')
        final = datetime.strptime(self.final_date, '%d-%m-%Y').strftime('%Y-%m-%d')
        pytrends = TrendReq(hl='pt-BR')
        pytrends.build_payload(self.app, timeframe=f'{inicio} {final}', geo=f'BR-{self.estado}')
        dicio = {}
        for i in self.app:
          dicio[i] = pytrends.related_queries()[i]['top'].rename(columns={'query':i})
          time.sleep(1)
        return dicio


  def graph(self, tabela):
        apps = self.app
        dicio = {}
        def normalize(x):
          min_max_scaler = preprocessing.MinMaxScaler()
          x_scaled = min_max_scaler.fit_transform(np.array(x).reshape(-1,1))
          novo_x = (pd.DataFrame(x_scaled) +1) ** 2
          return np.array(novo_x[0])
        node_colors=('blue', 'red')
        G = nx.Graph()
        # df_delivery = dicio['delivery'][:5]
        # ddicio[key][:5]

        # Cores das linhas
        for key in apps:
          G.add_node(key)
          for i, v in tabela[key][:5].iterrows():
            G.add_edge(key, v[key],color=node_colors[apps.index(key)], weigth=v['value'])

        edges = G.edges()
        colors = [G[u][v]['color'] for u, v in edges]
        # weights = [(G[u][v]['weigth'] / 50)**3 for u,v in edges]
        raw_weigths = np.array([G[u][v]['weigth'] for u, v in edges])
        weigths = normalize(raw_weigths)

        # Cores das bolinhas
        color_map = []
        for node in G:
          if node not in apps:
            color_map.append('#ccdcff')
          else:
            color_map.append(node_colors[apps.index(node)])

            # if node == apps[0]:
            #     color_map.append(node_colors[0])
            # elif node == apps[1]:
            #     color_map.append(node_colors[1])
            # elif node == apps[2]:
            #     color_map.append(node_colors[2])
            # else:
            #     color_map.append('#ccdcff')
        fig, ax = plt.subplots(figsize=(10,10))
        nx.draw_planar(G, with_labels=True, node_size=1000, width=list(weigths), edge_color = colors, node_color=color_map)
        return fig

def social_dataframe(lista_apps, country, estado, start_date, final_date, dicionario):
    '''
    Essa função executa o GeoEstimation().get_municip() e agrega os dados sobre PIB e população no arquivo ibge_municipios.xlsx.
    O retorno é um dataframe
    
    * Parâmetros:
    - lista_apps: a lista de aplicativos a serem analisados
    - country: a sigla do país
    - estado: a sigla do estado brasileiro
    - start_date: a data de início
    - final_date: a data final
    '''
    for nome_app in lista_apps:
        GeoEstimation(nome_app, country, start_date=start_date, final_date=final_date).get_municip_real(estado, dicionario)


    dado_lista = []
    for file in list(dicionario.keys()):
        if estado in file:
            print(file)
            dado_lista.append(pd.DataFrame(dicionario[file]))


    dado_agr = pd.concat(dado_lista)
    dado_agr['name_muni'] = dado_agr['name_muni'].str.title()
    dado_ibge = pd.read_excel('ibge_municipios_final.xlsx', engine='openpyxl')
    #dado_ibge['name_muni'] = dado_ibge['name_muni'].apply(lambda x: str(x).title())
    dado = dado_agr.merge(dado_ibge, how='left', on='name_muni')
    #dado.to_csv('agregado.csv')
    dado['geometry'] = dado['geometry'].astype('str').apply(wkt.loads) #Transformando a coluna geometry no tipo geometry
    dado = gpd.GeoDataFrame(dado, crs='epsg:4326') #Transformando o dataframe num geodataframe
    dado_filtrado = dado.groupby('name_muni').max().reset_index().iloc[:,:][lista_apps] #Criando um dataframe só com os apps como coluna
    dados_economicos = dado.groupby('name_muni').max().reset_index().iloc[:, -4:]
    dado_filtrado['app'] = dado_filtrado.idxmax(axis=1) #Pegando os nomes de coluna com maior valor e colocando numa nova coluna
    dado_filtrado.fillna(0, inplace=True) #Trocando nulo por 0
    #dado = dado.iloc[:len(dado_filtrado)] #Filtrando o dataframe
    dado_somado = pd.concat([dado.drop(columns=lista_apps), dado_filtrado], axis=1) #Juntando os dois dataframes
    dado_somado['soma'] = dado_somado[lista_apps].sum(axis=1) #Somando os valores
    dado_somado['app'] = np.where(dado_somado['soma'] == 0, 'Nenhum', dado_somado['app']) #Tirando os que repetiram erroneamente na coluna app
    dado_somado['max'] = dado_somado[lista_apps].max(axis=1) #Pegando o valor máximo
    #dado_somado = dado_somado.dropna(subset=['max']) #Excluir o nulo da coluna max

    dado_final = pd.concat([dado_somado, dados_economicos], axis=1).drop_duplicates(subset='code_muni').iloc[:,:-3].fillna(0)
    dado_final['geometry'] = dado_final['geometry'].astype(str).apply(wkt.loads)
    #dado_final.to_excel('dado_final.xlsx')
    return dado_final


def tendencia_mensal(apps_lista, estados_lista, lista_cores, data_inicial, data_final):
  df_lista_final3 = []
  base = datetime(2020,1,1)
  date_list = [base + relativedelta(months=x) for x in range(0,13,1)]
  app_color_dict = pd.DataFrame({'app':apps_lista, 'cor':lista_cores})
  for apps in apps_lista:
    lista_df = []
    for i,v in enumerate(date_list):
      if v != date_list[-1]:
        data_inicial = v.strftime('%d-%m-%Y')
        data_final = date_list[i+1].strftime('%d-%m-%Y')
        df = googletrends.spatio(apps, geo='BR', date_start=data_inicial,date_stop=data_final, method='')['BR']['df'].reset_index()
        df['data'] = data_inicial
        lista_df.append(df)
    df_final = pd.concat(lista_df, axis=1)

    print(df_final.columns)
    lista_estados = df_final['geoName'].iloc[:,0].tolist()
    df_final['estado'] = lista_estados
    df_final.drop(columns='geoName', inplace=True)

    lista_estados_selecionados = estados_lista
    df_lista2 = []
    for i in lista_estados_selecionados:
      df_manip = df_final[df_final['estado'] == i].set_index('data')
      df_lista2.append(df_manip)
    df_final2 = pd.concat(df_lista2)

    #df_final2.index = ['nubank']
    df_final2.columns = date_list
    df_final3 = df_final2.T
    df_final3 = df_final3.iloc[:-1,:]
    #df_final3['nubank'] = df_final3['nubank'].astype(int)
    df_final3.columns = lista_estados_selecionados
    df_final3['app'] = apps

    df_lista_final3.append(df_final3)
  df_final4 = pd.concat(df_lista_final3)
  return df_final4
  

def tendencia_mensal2(apps_lista, state, color, data_inicial, data_final):
  df_lista = []
  
  data_inicial = datetime.strptime(data_inicial, "%d-%m-%Y")
  data_final = datetime.strptime(data_final, "%d-%m-%Y")

  base = data_inicial
  
  r = relativedelta(data_final, data_inicial)
  
  date_list = date_list = [base + relativedelta(months=x) for x in range(0, r.months + (12*r.years))]
  pytrends = TrendReq(hl='pt-BR')
  for i,v in enumerate(date_list):
    if v != date_list[-1]:
      data_inicial = v.strftime('%Y-%m-%d')
      data_final = date_list[i+1].strftime('%Y-%m-%d')
      pytrends.build_payload(apps_lista, timeframe=f'{data_inicial} {data_final}', geo=f'BR')
      df = pytrends.interest_by_region(resolution='COUNTRY', inc_low_vol=True, inc_geo_code=True)
      df['data'] = data_inicial
      df_lista.append(df)
  df_final = pd.concat(df_lista).reset_index()

 
  df_final2 = df_final[df_final['geoCode'] == f'BR-{state}'].set_index('data')
  
  print(df_final2)
  return df_final2


def tendencia_brasil(apps_lista, state, data_inicial, data_final):
    data_inicial = datetime.strptime(data_inicial, '%d-%m-%Y').strftime('%Y-%m-%d')
    data_final = datetime.strptime(data_final, '%d-%m-%Y').strftime('%Y-%m-%d')
    pytrends = TrendReq(hl='pt-BR')
    pytrends.build_payload(apps_lista, timeframe=f'{data_inicial} {data_final}', geo=f'BR')
    #{data_inicial.split("-")[2]}-{data_inicial.split("-")[1]}-{data_inicial.split("-")[0]} {data_final.split("-")[2]}-{data_final.split("-")[1]}-{data_final.split("-")[0]}
    df = pytrends.interest_over_time().drop(columns='isPartial')
    return df
def get_municip_real(app, estado, start_date, final_date, dicionario):
        '''
        Essa função retorna os dados municipais dos Índices Google trends
        
        * Parâmetros:
        - estado: estado
        
        '''
        inicio = datetime.strptime(start_date, '%d-%m-%Y').strftime('%Y-%m-%d')
        final = datetime.strptime(final_date, '%d-%m-%Y').strftime('%Y-%m-%d')
        #df_brasil = GeoEstimation(self.app, self.country, start_date=self.start_date, final_date=self.final_date).dataframe()
        pytrends = TrendReq(hl='pt-BR')
        pytrends.build_payload(kw_list = app, timeframe=f'{inicio} {final}', geo=f'BR-{estado}')
        df_muni = interest_by_city(pytrends, inc_low_vol=True)
        #df_muni = df_muni[df_muni[self.app] >=1]
        #df_muni[f'{self.app}_taxa'] = df_muni[self.app] / df_muni[self.app].sum()
        
        df_muni = df_muni.reset_index()
        dado = df_muni
        #df_estado = df_brasil[df_brasil['abbrev_state'] == self.estado].reset_index()
        #print(df_brasil[df_brasil['abbrev_state'] == self.estado]['geo_rating_estimation'].values[0])
        '''
        df_final = pd.DataFrame()
        df_final['name_muni'] = df_muni['geoName'].str.title()
        df_final[self.app] = df_muni[self.app]
        dado_estado = geobr.read_municipality(code_muni='all', year=2020).query(f'abbrev_state == "{self.estado}"')
        #dado_estado = dado_estado[dado_estado['abbrev_state'] == self.estado]
        print(dado_estado)
        dado = dado_estado.merge(df_final, how='left', on='name_muni').fillna(0)
        '''
        return dado
#############################################################################


st.set_option('deprecation.showPyplotGlobalUse', False)


st.title('GeoLocation - MVP')
st.image("GeoLocation.png")
st.write('''Objetivo desse MVP: validar a ideia de estimar e analisar a geolocalização das buscas por apps e keywords.
         As buscas são baseadas no Google Trends, onde os termos recebem 100 para o local e data onde mais foram pesquisados, e 0 onde não houve volume relevante de busca''')
apps = st.text_input('Termos de busca: app1,app2,keyword1,keyword2')

app = apps.split(',')
pais = 'BR'
state = st.text_input('Estado (sigla)')

state_sigla = {'SP':'State of São Paulo',
'RJ':'State of Rio de Janeiro',
'MG':'State of Minas Gerais',
'ES':'State of Espírito Santo',
'AC':'State of Acre',
'AL':'State of Alagoas',
'AP':'State of Amapá',
'AM':'State of Amazonas',
'PR':'State of Paraná'
}

with urlopen('https://raw.githubusercontent.com/tbrugz/geodata-br/master/geojson/geojs-35-mun.json') as response:
 geojson_sp = json.load(response) # Javascrip object notation 

data_inicial = st.text_input('Data inicial: dia-mes-ano')
data_final = st.text_input('Data final: dia-mes-ano')

lista_cores = st.multiselect('Cores dos respectivos termos de busca:', ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds'])


app_color_dict = pd.DataFrame({'app':app, 'cor':lista_cores})
df_potencial = {'app':[],
'n_cidades':[],
'PIB_potencial':[],
'demanda_potencial':[],
'|':[],
'PIB_per_capita_medio':[],
'idh_medio':[],
'gini_medio':[]}

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def start_read():
    dicionario_arquivos = {}
    geo = GeoEstimation(app, pais, start_date=data_inicial, final_date=data_final)
    try:
     state_df = social_dataframe(app, 'BR', estado=state, start_date=data_inicial, final_date=data_final, dicionario = dicionario_arquivos)
    except:
      state_df = None
    return geo, state_df, dicionario_arquivos
#state_df = []

geo = start_read()[0]
state_df = start_read()[1]
dicionario_arquivos = start_read()[2]
df_municipios = get_municip_real(app = app, estado = state, start_date = data_inicial, final_date = data_final, dicionario=dicionario_arquivos)

st.markdown('## Análise no país')
if st.button('Exibir tabela (país)'):
    geo_pais_dataframe = geo.dataframe(dicionario_arquivos).set_index('abbrev_state').drop(columns='geometry')
    st.dataframe(geo_pais_dataframe)
if st.button('Exibir mapa (país)'):
    for row, value in app_color_dict.iterrows():
        st.pyplot(GeoEstimation(value['app'], pais, start_date=data_inicial, final_date=data_final).map(cor=value['cor'], dicionario = dicionario_arquivos))

st.markdown('## Análise no Estado')
if st.button('Exibir tabela (Estado)'):
    #st.text('visualizar')
    #st.text(state_df.columns)[['name_muni', 'abbrev_state', 'populacao', 'pib', 'IDH', 'pib_per_capita','app', 'soma', 'max']+app]
    #geo_state_dataframe = state_df[['name_muni', 'abbrev_state', 'app']+app]
    geo_state_dataframe = df_municipios
    for i in app:
      st.text(len(geo_state_dataframe[geo_state_dataframe[i] > 0]['geoName'].tolist()))
      st.text(geo_state_dataframe[geo_state_dataframe[i] > 0]['geoName'].tolist())
      st.text('')
    #geo_state_dataframe = state_df
    st.dataframe(geo_state_dataframe)
if st.button('Exibir mapa (Estado)'):
    for row, value in app_color_dict.iterrows():
        st.text(value['app'])
        st.pyplot(GeoEstimation(value['app'], 'BR', start_date=data_inicial, final_date=data_final).municip_map(state, cor = value['cor'], dicionario=dicionario_arquivos, geojson=geojson_sp))
if st.button('Estimativa socioeconômica'):
    df = state_df
    lista_app = app
    df_pib_pot = df[app]
    #[app+['pib','pib_per_capita', 'populacao','IDH','gini']]
    '''
    #for i in app:
     # st.text(i)
      #st.text(df[df[i] > 0].sort_values(by=i,ascending=False)['name_muni'])
    for i in lista_app:
      df_pib_pot[f'taxa_{i}'] = (df_pib_pot[i] / df_pib_pot[lista_app].sum(axis=1)) * (df_pib_pot[i] /100)
    df_pib_pot = df_pib_pot.dropna(0)
    for i in lista_app:
      print(df_pib_pot.columns)
      df_pib_pot[f'taxa_pib_{i}'] = df_pib_pot['pib'] * df_pib_pot[f'taxa_{i}']
      df_pib_pot = df_pib_pot.iloc[:,~df_pib_pot.columns.duplicated()]
      df_pib_pot[f'pib_per_capita_medio_{i}'] = df_pib_pot[df_pib_pot[i] > 0]['pib_per_capita']
      df_pib_pot[f'taxa_demanda_{i}'] = round(df_pib_pot['populacao'] * df_pib_pot[f'taxa_{i}'], 0)
      df_pib_pot[f'taxa_idh_{i}'] = df_pib_pot[df_pib_pot[i] > 0]['IDH']
      df_pib_pot[f'taxa_gini_{i}'] = df_pib_pot[df_pib_pot[i] > 0]['gini']
      

    for i in lista_app:
      df_potencial['app'].append(i)
      df_potencial['n_cidades'].append(len(df_pib_pot[df_pib_pot[i] > 0]))
      df_potencial['PIB_potencial'].append(round(df_pib_pot[f'taxa_pib_{i}'].sum(),2))
      df_potencial['demanda_potencial'].append(round(df_pib_pot[f'taxa_demanda_{i}'].sum(),0))
      df_potencial['|'].append('|')
      df_potencial['PIB_per_capita_medio'].append(round(df_pib_pot[f'pib_per_capita_medio_{i}'].mean(),2))
      df_potencial['idh_medio'].append(df_pib_pot[f'taxa_idh_{i}'].mean())
      df_potencial['gini_medio'].append(df_pib_pot[f'taxa_gini_{i}'].mean())'''
    st.dataframe(df_pib_pot)

#input_estados_tendencia = st.text_input('Estados para comparar a tendência mensal')
sigla_estado = state_sigla[state]
select_estado = [sigla_estado]

st.markdown('## Análise de tendência')
botao_tendencia = st.button('Tendência')
if botao_tendencia:
    st.markdown('O primeiro gráfico representa o quanto cada termo representou para o Estado a cada mês, sendo a soma de cada mês igual a 100 (podemos, então, interpretar como percentual)')
    st.markdown('O segundo gráfico representa a evolução temporal de cada termo para o Brasil, semanalmente')
    adapt_cores = [i.lower()[:-1] for i in lista_cores]
    app_color_dict['cor'] = adapt_cores
    retorno_df = tendencia_mensal2(app, state, adapt_cores, data_inicial = data_inicial, data_final = data_final)
    #retorno_df1 = tendencia_mensal(app, state, adapt_cores, data_inicial = data_inicial, data_final = data_final)
    #estados_lista= [select_estado]
    #estado = select_estado
    fig, ax = plt.subplots(1,1, figsize=(10, 6), dpi=200)
    retorno_df[app_color_dict['app']].plot(ax=ax,label=app_color_dict['app'], color=[i[:-1] if i[-1] == 's' else i for i in app_color_dict['cor']])
    
    #df_final4[df_final4['app'] == 'picpay'][[estado,'app']].plot(ax=ax, label='picpay',color='green')
    #df_final4[df_final4['app'] == 'c6 bank'][[estado,'app']].plot(ax=ax, label='c6 bank', color='black')
    #df_final4[df_final4['app'] == 'banco inter'][[estado,'app']].plot(ax=ax, label='banco inter',color='orange')
    plt.legend(app)
    plt.rcParams.update({"font.size": 10})
    plt.ylim(0,100)
    #plt.xticks(fontsize=8, rotation=45)
    plt.title(f'Proporção mensal ({retorno_df["geoName"][0]}, {data_inicial.split("-")[2]})', fontsize=15)
    plt.ylabel('Google Trends Index')
    plt.box(False)
    st.pyplot(fig)
    
    
    fig2, ax2 =plt.subplots(1,1, figsize=(10, 6), dpi=200)   
    brasil_df = tendencia_brasil(app, state, data_inicial=data_inicial, data_final=data_final)
    #brasil_df = brasil_df[brasil_df['Km de vantagens'] < 90]
    brasil_df[app_color_dict['app']].plot(ax=ax2,label=app_color_dict['app'], color=[i[:-1] if i[-1] == 's' else i for i in app_color_dict['cor']])
    plt.rcParams.update({"font.size": 10})
    plt.title(f'Evolução semanal no Brasil, {data_inicial.split("-")[2]}', fontsize=15)
    plt.box(False)
    plt.xlabel('data')
    plt.ylabel('Google Trends Index')
    plt.ylim(0,100)
    st.pyplot(fig2)




#variavel_socio = st.selectbox('Escolha uma variável para correlacionar com os termos de busca')

st.markdown('## Análise de clusters espaciais')

if st.button('Clusters univariados'):
    st.text('''
    HH: Municípios com valores altos, que são vizinhos de municípios com valores altos
    HL: Municípios com valores altos, que são vizinhos de municípios com valores baixos
    LH: Municípios com valores baixos, que são vizinhos de municípios com valores altos
    LL: Municípios com valores baixos, que são vizinhos de municípios com valores baixos
    ns: Sem significância estatística, ou seja, provavelmente a relação de vizinhança é aleatória (sem explicação)
    ''')
    dicio_cluster = {}
    set_list = []
    for j in app:
        try:
          dado = read_geodata(dicionario_arquivos[f'{j}_{state}_with_geometry'])
          #dado = state_df
          i_moran = otimizar_k(dado, j, 1, 10, p_value=0.05)
          pesos = weights_matrix(dado, metric = 'knn', k = i_moran)
          st.pyplot(plot_lisa(dado, j, weights= pesos, k_opt=i_moran, estado=state))
          st.text(f'Municípios HH para {j}: {", ".join(significant_HH(dado, j, weight= pesos))}')
          dicio_cluster.update({j:significant_HH(dado, j, weight= pesos)})
          set_list.append(set(dicio_cluster[j]))
        except:
          pass
    try:
      st.markdown(f'Interseção dos clusters HH:')
      st.text(set.intersection(*set_list))

      st.markdown('União dos clusters HH:')
      st.text(set.union(*set_list))
    except:
      st.text('Não foram encontrados clusters')
    
st.markdown('## Análise de pesquisas relacionadas')
if st.button('Pesquisas relacionadas'):
    tabelas = geo.similar_keywords(state)
    for i in app:
        st.dataframe(tabelas[i])
    st.pyplot(geo.graph(tabela=tabelas))
