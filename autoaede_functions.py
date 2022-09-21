import matplotlib.pyplot as plt  # Graphics
from matplotlib import colors
import seaborn as sns            # Graphics
import geopandas                 # Spatial data manipulation
import pandas as pd              # Tabular data manipulation
from pysal.explore import esda   # Exploratory Spatial analytics
from pysal.lib import weights    # Spatial weights
import contextily
import json
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
import openpyxl
import geobr
from shapely import wkt
from sklearn.preprocessing import StandardScaler
from splot import esda as esdaplot

def read_geodata(dados):
    # estados = geobr.read_state(year=2017)
    # estado = geobr.read_municipality(code_muni='RJ', year=2020)
    df = pd.DataFrame(dados)
    #df = gpd.GeoDataFrame(df, geometry='geometry')
    df['geometry'] = df['geometry'].astype(str).apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, crs='epsg:4326')


    # df = df.merge(estado, how='left', on='code_muni')
    return gdf
    
def otimizar_k(tabela, coluna, k_min, k_max, p_value=0.05):
  '''Essa função otimiza o valor de K para a matriz de pesos espaciais
  baseada nos k-viznhos mais próximos.
  
  -- Parâmetros:
  * tabela: o dataframe contendo a variável
  * coluna: a variável que será utilizada como peso na matriz
  * k: o número máximo de iterações em k

  -- Retornos da função:
  * .valores: retorna uma tabela contendo o resultado das iterações em k com
  ** "k", "moran_index" e "p_value"
  * melhor_k: retorna o valor de k com a maior variação percentual
  '''
  i_list = [0]
  p_list = [0]
  k_list = [0]
  for j in range(k_min,k_max+1):
    w = weights.KNN.from_dataframe(tabela, k=j)
    w.transform='R'
    moran = esda.moran.Moran(tabela[coluna],w)
    i_list.append(moran.I)
    p_list.append(moran.p_sim)
    k_list.append(j)
    valores = pd.DataFrame({'moran_index':i_list,
              'p_value':p_list,
              'k':k_list}, index = k_list).sort_values('k')
  return valores[valores['k'] > 0].set_index('k').query(f'p_value <= {p_value}')['moran_index'].idxmax()
    
def weights_matrix(dados, k, metric='rainha'):
    # Calculate Weights
    if metric == 'rainha':
        weight = weights.contiguity.Queen.from_dataframe(dados)
    elif metric == 'torre':
        weight = weights.contiguity.Rook.from_dataframe(dados)
    elif metric == 'knn':
        weight = weights.KNN.from_dataframe(dados, k=k)  
    return weight


def plot_weights(dados, weight):
    f, axs = plt.subplots(1, 1, figsize=(15, 8))
    # Plot map
    ax = dados.plot(edgecolor='k', facecolor='w', ax=axs)
    # Plot graph connections
    weight.plot(
        dados, ax=axs,
        edge_kws=dict(color='r', linestyle=':', linewidth=2),
        node_kws=dict(marker='')
    )
    plt.box(False)
    plt.title('Mapa de vizinhança dos municípios do Estado do Rio de Janeiro', fontfamily='serif', fontsize=20)
    plt.xticks(fontsize=0)
    plt.yticks(fontsize=0)

    #f.savefig(f'plot_weights_{metric}.png')
    return f

def plot_moran(tabela, coluna, weight):
  # Row-standardization
  dados_geo = tabela
  w = weight
  w.transform = 'R'
  dados_geo[f'{coluna}_norm'] = StandardScaler().fit_transform(np.array(dados_geo[coluna]).reshape(-1,1))
  dados_geo[f'{coluna}_W'] = weights.spatial_lag.lag_spatial(w, dados_geo[f'{coluna}'])
  dados_geo[f'{coluna}_norm_W'] = weights.spatial_lag.lag_spatial(w, dados_geo[f'{coluna}_norm'])
  dados_geo[f'{coluna}_std'] = (dados_geo[coluna] - dados_geo[coluna].mean())
  dados_geo[f'{coluna}_norm_W_std'] = (dados_geo[f'{coluna}_norm_W'] - dados_geo[f'{coluna}_norm_W'].mean())

  moran = esda.moran.Moran(dados_geo[coluna],w)
  f, ax = plt.subplots(1, figsize=(10, 10))
  sns.regplot(x=f'{coluna}_norm', y=f'{coluna}_norm_W', 
                  ci=None, data=dados_geo, line_kws={'color':'b'}, scatter_kws={'s':20})
  ax.axvline(dados_geo[f'{coluna}_norm'].mean(), c='k', alpha=0.5)
  ax.axhline(dados_geo[f'{coluna}_norm_W'].mean(), c='k', alpha=0.5)
  plt.suptitle(f'Moran Index = {round(moran.I,4)}',fontsize=20, fontfamily='serif')
  # for x,y,s in zip(dados_geo[f'{coluna}_norm'] ,dados_geo[f'{coluna}_norm_W'],dados_geo['municipio_nome']):
  #   plt.annotate(s=s,xy=(x-0.003,y-0.006), fontsize=12)
  plt.title(f'p-value = {moran.p_sim}',fontsize=16, fontfamily='serif')
  plt.annotate('HH', xy=(3,2), fontsize=25, color='red', fontfamily='serif')
  plt.annotate('LL', xy=(-2,-1), fontsize=25, color='red', fontfamily='serif')
  plt.annotate('HL', xy=(3,-1), fontsize=25, color='red', fontfamily='serif')
  plt.annotate('LH', xy=(-2,2), fontsize=25, color='red', fontfamily='serif')
  
  f.savefig('plot_moran.png')
  return f

def map_weighted(tabela, coluna, titulo):
    dados_geo = tabela
    fig, axs = plt.subplots(1,2, figsize=(30,30))

    fig.suptitle(titulo, fontsize=30, fontfamily='serif', y=0.7)

    dados_geo.plot(column=coluna,  edgecolor='black', cmap='Blues', ax=axs[0])
    axs[0].set_title('Sem ponderação espacial', fontsize=25, fontfamily='serif')
    axs[0].axis('off')


    dados_geo.plot(column=f'{coluna}_W',  edgecolor='black', cmap='Blues', ax=axs[1])
    axs[1].set_title('Com ponderação espacial', fontsize=25, fontfamily='serif')
    axs[1].axis('off')

    fig.savefig('maps_weighted.png')
    return fig

def plot_lisa(tabela, coluna, weights, k_opt, estado):
    w = weights
    dados_geo = tabela
    lisa = esda.moran.Moran_Local(dados_geo[coluna], w)
    # Set up figure and axes
    f, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    f.suptitle(f'Clusters de {coluna} em {estado}', fontsize=20, fontfamily='serif')
    #      
                        # Subplot 4 #
                        # Cluster map
    # Grab second axis of local statistics
    #ax = axs[3]
    # Plot Quandrant colors In this case, we use a 5% significance
    # level to select polygons as part of statistically significant
    # clusters
    esdaplot.lisa_cluster(lisa, dados_geo, p=0.05, ax=axs);

                        # Figure styling #
    # Set title to each subplot
    """for i, ax in enumerate(axs.flatten()):
        ax.set_axis_off()
        ax.set_title(
            [
                'Moran Cluster Map'
            ][i], y=0, fontfamily='serif'
        )"""
    # Tight layout to minimise in-betwee white space
    f.tight_layout()
    #f.savefig('LISA maps.png')
    return f    


def significant_HH(dados_geo, coluna, weight):
    w = weight
    w.transform = 'R'
    dados_geo[f'{coluna}_norm'] = StandardScaler().fit_transform(np.array(dados_geo[coluna]).reshape(-1,1))
    dados_geo[f'{coluna}_W'] = weights.spatial_lag.lag_spatial(w, dados_geo[f'{coluna}'])
    dados_geo[f'{coluna}_norm_W'] = weights.spatial_lag.lag_spatial(w, dados_geo[f'{coluna}_norm'])
    dados_geo[f'{coluna}_std'] = (dados_geo[coluna] - dados_geo[coluna].mean())
    dados_geo[f'{coluna}_norm_W_std'] = (dados_geo[f'{coluna}_norm_W'] - dados_geo[f'{coluna}_norm_W'].mean())
    
    lisa = esda.moran.Moran_Local(dados_geo[coluna], w)
    labels = pd.Series(
    1 * (lisa.p_sim < 0.05), # Assign 1 if significant, 0 otherwise
    index=dados_geo.index           # Use the index in the original data
    # Recode 1 to "Significant and 0 to "Non-significant"
    ).map({1: 'Significant', 0: 'Non-Significant'})

    df_sig = pd.concat([dados_geo,labels], axis=1)
    return df_sig[(df_sig[0] == 'Significant') & (df_sig[f'{coluna}_norm'] > 0) & (df_sig[f'{coluna}_norm_W'] > 0)]['name_muni'].tolist()
    

"""

app = 'nubank'
dado = read_geodata(f'{app}_SP_with_geometry.csv')

i_moran = otimizar_k(dado, 'nubank', 1, 10, p_value=0.05)

print('Melhor I de Moran:', i_moran)

pesos = weights_matrix(dado, metric = 'knn', k = i_moran)

plot_weights(dado, pesos)

plot_moran(dado, app, weight=pesos)

map_weighted(dado, app, f'Pesquisa por {app.capitalize()} em SP')

plot_lisa(dado, app, weights= pesos, k_opt=i_moran)"""
