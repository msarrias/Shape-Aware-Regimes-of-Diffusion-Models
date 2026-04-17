import numpy as np
from sklearn.manifold import TSNE
from construct_graph import construct_graph
from graph_distance import get_symbiharmonic_coords
import time
from adaptive_knn import AdaptiveKNNGraph


def SASNE(data, n_components=2):
    start_time = time.time()
    print('Constructing graph...')
    obj_knn = AdaptiveKNNGraph(data)
    W = obj_knn.compute_W() #construct_graph(data)
    print("--- %s seconds elapsed ---" % round(time.time() - start_time,5))
    start_time = time.time()
    print('Computing graph distance...')
    res = get_symbiharmonic_coords(W)
    print("--- %s seconds elapsed ---" % round(time.time() - start_time,5))
    Z, eigenval = res
    init_Y = 1e-4 * Z[:,[1,2]] * np.sqrt(eigenval[1])
    n = len(W)
    perplexity = 0.9 * n
    start_time = time.time()
    print('Computing embedding...')
    embedding = TSNE(n_components=n_components,init=init_Y,perplexity=perplexity).fit_transform(Z)
    print("--- %s seconds elapsed ---" % round(time.time() - start_time,5))   

    return embedding,Z


