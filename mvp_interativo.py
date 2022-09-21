import streamlit as st
from osmnx_functions import osmnx
from streamlit_folium import st_folium
import folium
import geopandas




#get_gdf, opcoes_unicas, interactive_map_by_amenity, interactive_map_by_name



st.title('MVP - Interactive GeoLocation')

osmnx = osmnx()

local = st.text_input('Local')

tipo = st.selectbox('Selecione a segmentação (amenity= tipo de comércio / name = nome do comércio)', ['amenity', 'name'])

dataframe = osmnx.get_gdf(local)

#st.table(dataframe.drop(columns=['geometry']))

lista_opcoes = osmnx.opcoes_unicas(tipo)

lista_opcoes_selecionadas = st.multiselect('Selecione seu filtro', lista_opcoes)





if len(lista_opcoes_selecionadas) <= 3:
    gdf = dataframe
    colors = ['red','blues','green']
    m = gdf[gdf[tipo].isin(lista_opcoes_selecionadas)].explore(tooltip=['amenity','name','addr:city','addr:suburb','addr:street'])


    for i,v in enumerate(lista_opcoes_selecionadas):
        geopandas.GeoDataFrame(gdf[gdf[tipo] == v][['amenity','name','addr:city','addr:suburb','addr:street','geometry']]).explore(
            m=m, # pass the map object
            color=colors[i], # use red color on all points
            marker_kwds=dict(radius=5, fill=True), # make marker radius 10px with fill
            tooltip=['amenity','name','addr:city','addr:suburb','addr:street'], # show "name" column in the tooltip
            tooltip_kwds=dict(labels=True), # do not show column label in the tooltip
            name="cities" # name of the layer in the map
        )


    folium.TileLayer('Stamen Toner', control=True).add_to(m)  # use folium to add alternative tiles
    folium.LayerControl().add_to(m)  # use folium to add layer control

    st_folium(m, width=1000, height=700)
else:
    st.text('Selecione até 3 opcoes')
