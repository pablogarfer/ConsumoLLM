import time

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.cluster import MeanShift, DBSCAN
from sklearn import metrics
from sklearn.impute import KNNImputer
from math import floor
import seaborn as sns


def norm_to_zero_one(df):
    return (df - df.min()) * 1.0 / (df.max() - df.min())
datos = pd.read_csv('clustering/emissions.csv', sep=';', decimal=',')

# Se imputa por vecinos más cercanos y se normaliza a [0,1]
#'''
imputer = KNNImputer(n_neighbors=3)
datos_norm = datos.apply(norm_to_zero_one)
datos_imputados_array = imputer.fit_transform(datos_norm)
datos_norm2 = pd.DataFrame(datos_imputados_array, columns = datos.columns)
datos = datos_norm2
#'''

# Seleccionar variables de interés para clustering
subset=datos.copy()
usadas = ['Duracion', 'Energia Total','Energia CPU', 'Energia GPU', 'Energia RAM', 'Tokens', 'Parametros']
n_var = len(usadas)
X = subset[usadas]
plot_num = 1

# normalizamos
X_normal = X.apply(norm_to_zero_one)

ms = MeanShift()
dbscan = DBSCAN()

#Definimos los algoritmos que vamos a probar

clustering_algorithms = (
        ("MeanShift", ms),
        ("DBSCAN", dbscan),
    )

dict_list = []

#Para cada algoritmo vamos incrementando su bandwidth o epsilon

for name, algorithm in clustering_algorithms:
    k = 0.05

    #Se limita en 0.7 ya que alrededor de esa medida siempre se genera 1 solo cluster
    while(k <= 0.7):

        if(name == 'MeanShift'):
            ms = MeanShift(bandwidth=k, bin_seeding=True)
            cluster_predict = ms.fit_predict(X_normal)
        elif(name== 'DBSCAN'):
            dbscan = DBSCAN(eps=k)
            cluster_predict = dbscan.fit_predict(X_normal) 

        #Comprobamos si hay más de un cluster, en caso de que solo haya 1 no se calculan las medidasb y se añaden
        n_clusters = np.unique(cluster_predict)  
        if(n_clusters.size != 1):     
            muestra_silhouette = 0.2 if (len(X) > 10000) else 1.0
            metric_SC = metrics.silhouette_score(X_normal, cluster_predict, metric='euclidean', sample_size=floor(muestra_silhouette*len(X)), random_state=123456)

            metric_CH = metrics.calinski_harabasz_score(X_normal, cluster_predict)

            row_dict = {'Algoritmo': name, 'CH Index': metric_CH, 'Silhouette': metric_SC, 'K': k, 'Clusters': n_clusters.size}
            dict_list.append(row_dict)
      
        k+= 0.01   

data = pd.DataFrame(columns = ['Algoritmo', 'CH Index', 'Silhouette', 'K', 'Clusters'])
data = pd.DataFrame.from_dict(dict_list)
data.to_excel(f"clustering/caso1/output.xlsx")

f, axs = plt.subplots(2,1, sharex=True)
sns.lineplot(x="K", y= "Silhouette", hue = "Algoritmo" , data=data, style = "Algoritmo", ax=axs[0])
sns.lineplot(x="K", y= "CH Index", hue = "Algoritmo" , data=data, style = "Algoritmo", ax=axs[1])
f.tight_layout()
f.savefig(f"clustering/caso1/metrics.png")
    