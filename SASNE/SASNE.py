import numpy as np
from sklearn.manifold import TSNE
from graph_distance import get_symbiharmonic_coords
import time
from adaptive_knn import AdaptiveKNNGraph


def SASNE(data, n_components=2):
    obj_knn = AdaptiveKNNGraph(data)
    W = obj_knn.compute_W() #construct_graph(data)
    res = get_symbiharmonic_coords(W)
    Z, eigenval = res
    init_Y = 1e-4 * Z[:,[1,2]] * np.sqrt(eigenval[1])
    n = len(W)
    perplexity = 0.9 * n
    embedding = TSNE(
        n_components=n_components,
        init=init_Y,
        perplexity=perplexity
    ).fit_transform(Z)

    return embedding, Z


