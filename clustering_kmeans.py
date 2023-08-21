from sklearn.cluster import KMeans
import numpy as np

def clustering_kmeans(data,maxK,error_factor=0.1):
    '''  
    @maxK: maximo nro de clusters
    @error_factor: tasa de mejora de error limite aceptable.
    
    '''
    
    #Cluster con mejor calidad
    maxClusterTest = maxK
    metrica = []
    metrica.append(999999)
    best_k = 1 #minimo
    i = 1
    kmeans_ant = ''
    for k in range(3,maxClusterTest):

        kmeans = KMeans(k,algorithm="full",max_iter=1000000, random_state = 120)
        kmeans.fit(data)
        metrica.append(kmeans.inertia_)

        mejora = np.abs(np.divide(metrica[i]-metrica[i-1],metrica[i-1]))

        if mejora > error_factor:
            best_k = k
            kmeans_ant = kmeans
        else:
            break

        i+=1
    return best_k,kmeans_ant