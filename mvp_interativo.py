import streamlit as st
from osmnx_functions import osmnx
from streamlit_folium import st_folium
import folium
import pandas as pd
import geopandas
from shapely.geometry import Point
import numpy as np
import warnings
warnings.filterwarnings("ignore")


#get_gdf, opcoes_unicas, interactive_map_by_amenity, interactive_map_by_name



st.title('MVP - Interactive GeoLocation')

osmnx = osmnx()

local = st.text_input('Local')

tipo = st.selectbox('Selecione a segmentação (amenity= tipo de comércio / name = nome do comércio)', ['amenity', 'name'])

dataframe = osmnx.get_gdf(local)

#st.table(dataframe.drop(columns=['geometry']))

lista_opcoes = osmnx.opcoes_unicas(tipo)

lista_opcoes_selecionadas = st.multiselect('Selecione seu filtro', lista_opcoes)


df_localidade = dataframe[dataframe[tipo].isin(lista_opcoes_selecionadas)]

arquivo_input = st.file_uploader('Dado de relogios em csv')
if arquivo_input:
  df_relogio = pd.read_csv(arquivo_input)
st.write(df_relogio)
df_relogio['geometry'] = [Point(xy) for xy in zip(df_relogio['lng'], df_relogio['lat'])]
df_relogio = geopandas.GeoDataFrame(df_relogio)

dist_dict = {}
for i, v in df_relogio.iterrows():
  dist_list = []
  for j in range(len(df_localidade)):
    dist = np.linalg.norm(np.array([df_relogio['geometry'][i].x, df_relogio['geometry'][i].y]) - np.array([df_localidade['geometry'][j].x, df_localidade['geometry'][j].y]))
    #dist = distance.euclidean([df['geometry'][0].x, df['geometry'][0].y] , [farmacia_filtro['geometry'][i].x, farmacia_filtro['geometry'][i].y])
    dist_list.append(dist)
  dist_dict.update({v['ownerScreenId']: np.median(dist_list)})


dado_distancia = pd.DataFrame(dist_dict, index=['distancia_media']).T


if len(lista_opcoes_selecionadas) <= 3:
    gdf = dataframe
    colors = ['red','blue','green']
    m = df_localidade.explore(tooltip=['amenity','name','addr:city','addr:suburb','addr:street'])


    for i,v in enumerate(lista_opcoes_selecionadas):
        geopandas.GeoDataFrame(gdf[gdf[tipo] == v][['amenity','name','addr:city','addr:suburb','addr:street','geometry']]).explore(
            m=m, # pass the map object
            color=colors[i], # use red color on all points
            marker_kwds=dict(radius=5, fill=True), # make marker radius 10px with fill
            tooltip=['amenity','name','addr:city','addr:suburb','addr:street'], # show "name" column in the tooltip
            tooltip_kwds=dict(labels=True), # do not show column label in the tooltip
            name="cities" # name of the layer in the map
        )

    df_relogio.merge(dado_distancia.reset_index(), left_on=['ownerScreenId'], right_on='index').drop(columns='index').explore(
         m=m, # pass the map object
         color="blue", # use red color on all points
         marker_kwds=dict(radius=5, fill=True), # make marker radius 10px with fill
         tooltip=['address','ownerScreenId','distancia_media'], # show "name" column in the tooltip
         tooltip_kwds=dict(labels=True), # do not show column label in the tooltip
         name="cities" # name of the layer in the map
     )

    df_relogio.merge(dado_distancia.reset_index(), left_on=['ownerScreenId'], right_on='index').drop(columns='index').sort_values('distancia_media', ascending=True)[:10].explore(
         m=m, # pass the map object
         color="green", # use red color on all points
         marker_kwds=dict(radius=5, fill=True), # make marker radius 10px with fill
         tooltip=['address','ownerScreenId','distancia_media'], # show "name" column in the tooltip
         tooltip_kwds=dict(labels=True), # do not show column label in the tooltip
         name="cities" # name of the layer in the map
    )
    
    folium.TileLayer('Stamen Toner', control=True).add_to(m)  # use folium to add alternative tiles
    folium.LayerControl().add_to(m)  # use folium to add layer control

    st_folium(m, width=1000, height=700)
else:
    st.text('Selecione até 3 opcoes')
