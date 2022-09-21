import osmnx as ox
import geobr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import folium
import mapclassify
import geopandas



class osmnx:

    def __init__(self):
        pass
        
    def get_gdf(self, local):
      gdf = ox.geometries_from_place(local, {'amenity':True})
      gdf['name'] = gdf['name'].str.replace("'",'')
      gdf['geometry'] = gdf['geometry'].centroid
      self.gdf = gdf
      return gdf
      
      
    def opcoes_unicas(self, coluna):
      if coluna == 'amenity':
        return self.gdf['amenity'].unique().tolist()
      elif coluna == 'name':
        return self.gdf['name'].unique().tolist()
        
        
    def interactive_map_by_amenity(self, lista_amenity):
      colors = ['red','blues','green']
      m = self.gdf[self.gdf['amenity'].isin(lista_amenity)].explore(tooltip=['amenity','name','addr:city','addr:suburb','addr:street'])


      for i,v in enumerate(lista_amenity):
        geopandas.GeoDataFrame(self.gdf[self.gdf['amenity'] == v][['amenity','name','addr:city','addr:suburb','addr:street','geometry']]).explore(
            m=m, # pass the map object
            color=colors[i], # use red color on all points
            marker_kwds=dict(radius=5, fill=True), # make marker radius 10px with fill
            tooltip=['amenity','name','addr:city','addr:suburb','addr:street'], # show "name" column in the tooltip
            tooltip_kwds=dict(labels=True), # do not show column label in the tooltip
            name="cities" # name of the layer in the map
        )


      folium.TileLayer('Stamen Toner', control=True).add_to(m)  # use folium to add alternative tiles
      folium.LayerControl().add_to(m)  # use folium to add layer control

      return m


    def interactive_map_by_name(self, lista_name):
      colors = ['red','blues','green']
      m = self.gdf[self.gdf['name'].isin(lista_name)].explore(tooltip=['amenity','name','addr:city','addr:suburb','addr:street'])


      for i,v in enumerate(lista_name):
        geopandas.GeoDataFrame(self.gdf[self.gdf['name'] == v][['amenity','name','addr:city','addr:suburb','addr:street','geometry']]).explore(
            m=m, # pass the map object
            color=colors[i], # use red color on all points
            marker_kwds=dict(radius=5, fill=True), # make marker radius 10px with fill
            tooltip=['amenity','name','addr:city','addr:suburb','addr:street'], # show "name" column in the tooltip
            tooltip_kwds=dict(labels=True), # do not show column label in the tooltip
            name="cities" # name of the layer in the map
        )


      folium.TileLayer('Stamen Toner', control=True).add_to(m)  # use folium to add alternative tiles
      folium.LayerControl().add_to(m)  # use folium to add layer control

      return m