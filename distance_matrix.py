import numpy as np
from tqdm import tqdm
from SAGD import SAGD

def construct_sagd_distance_matrix(
    W_list,
    laplacian_type="unnormalized",
    norm_type="norm_wrt_avg_ctd"
):
    """
    Computes the pairwise SAGD distance between all graphs in W_list.
    Returns an (M x M) distance matrix where M is the number of graphs.
    """
    num_graphs = len(W_list)
    sagd_dist_matrix = np.zeros((num_graphs, num_graphs))

    for i in tqdm(range(num_graphs), desc="Computing SAGD Matrix"):
        for j in range(i + 1, num_graphs):
            dist = SAGD(
                W_i=W_list[i], 
                W_j=W_list[j], 
                laplacian_type=laplacian_type, 
                norm_type=norm_type
            )
            sagd_dist_matrix[i, j] = dist
            sagd_dist_matrix[j, i] = dist # symmetry
            
    return sagd_dist_matrix

