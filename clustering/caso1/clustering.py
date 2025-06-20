# -*- coding: utf-8 -*-

import time

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans, MeanShift, DBSCAN
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

subset=datos.copy()
usadas = ['Duracion', 'Energia Total','Energia CPU', 'Energia GPU', 'Energia RAM', 'Tokens', 'Parametros']

n_var = len(usadas)
X = subset[usadas]
plot_num = 1

# normalizamos
X_normal = X.apply(norm_to_zero_one)

#Inicializamos los algoritmos con sus parámetros

n_clusters = 3
bandwidth = 0.6
eps = 0.45

kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=5, random_state=123456)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
dbscan = DBSCAN(eps=eps)

#Asociamos el nombre de los algoritmos con el objeto para poder iterarlos

clustering_algorithms = (
        ("KMeans", kmeans),
        ("MeanShift", ms),
        ("DBSCAN", dbscan),
    )

#Iteramos por algoritmo para sacar las gráficas

for name, algorithm in clustering_algorithms:
    
    print(f"----- Ejecutando {name}", end='')
    t = time.time()
    cluster_predict = algorithm.fit_predict(X_normal)
    tiempo = time.time() - t
    print(": {:.2f} segundos, ".format(tiempo), end='')
    
    metric_CH = metrics.calinski_harabasz_score(X_normal, cluster_predict)
    print("Calinski-Harabasz Index: {:.3f}, ".format(metric_CH), end='')

    # Esto es opcional, el cálculo de Silhouette puede consumir mucha RAM.
    # Si son muchos datos, digamos más de 10k, se puede seleccionar una muestra, p.ej., el 20%
    muestra_silhouette = 0.2 if (len(X) > 10000) else 1.0
    
    metric_SC = metrics.silhouette_score(X_normal, cluster_predict, metric='euclidean', sample_size=floor(muestra_silhouette*len(X)), random_state=123456)
    print("Silhouette Coefficient: {:.5f}".format(metric_SC))
    
    # se convierte la asignación de clusters a DataFrame
    clusters = pd.DataFrame(cluster_predict,index=X.index,columns=['cluster'])   
    
    size=clusters['cluster'].value_counts()
    size = size.sort_index()
    
    for i,c in enumerate(size):
        print('%s: %5d (%5.2f%%)' % (i,c,100*c/len(clusters)))
    
    k = len(size)
    colors = sns.color_palette(palette='Set2', n_colors=k, desat=None)
    
    # se añade la asignación de clusters como columna a X
    X_algorithm = pd.concat([X, clusters], axis=1)
    #'''
    print("---------- Scatter matrix...")
    plt.figure()
    sns.set()
    variables = list(X_algorithm)
    variables.remove('cluster')
    X_algorithm['cluster'] += 1
    sns_plot = sns.pairplot(X_algorithm, vars=variables, hue="cluster", palette=colors, plot_kws={"s": 25}, diag_kind="hist", diag_kws={'multiple': 'stack'}) #en hue indicamos que la columna 'cluster' define los colores
    X_algorithm['cluster'] -= 1
    sns_plot.fig.subplots_adjust(wspace=.03, hspace=.03)
    sns_plot.fig.set_size_inches(15,15)
    sns_plot.savefig(f"clustering/caso1/{name}.png")