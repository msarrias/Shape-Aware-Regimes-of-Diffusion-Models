from scipy.spatial.distance import pdist, squareform
import numpy as np


class AdaptiveKNNGraph:
    def __init__(
            self,
            data: np.ndarray,
            min_k: int = 5,
            inject_edges=True,
            perc=0.02,
            kernel='gaussian'
    ):
        self.data = data
        self.min_k = min_k
        self.dist_matrix = squareform(pdist(data, metric='euclidean'))
        self.n_samples = len(self.dist_matrix)
        self.inject_edges = inject_edges
        self.kernel = kernel
        if self.inject_edges:
            self.true_dist_matrix = self.dist_matrix.copy()
            self.perc = perc
            self.inject_random_edges()
        else:
            self.true_dist_matrix = self.dist_matrix
        self.sorted_ind = np.argsort(self.dist_matrix, axis=0)
        self.k = None
        self.sigma = None

    def inject_random_edges(
            self
    ):
        num_nodes = int(self.n_samples * self.perc)
        pairs = np.random.permutation(self.n_samples)[:num_nodes].reshape(-1, 2)

        for vi, vj in pairs:
            self.dist_matrix[vi, vj] = 0.0
            self.dist_matrix[vj, vi] = 0.0

    def _depth_first_search(
            self,
            v: int,
            marked: set,
            unmarked: set,
            A: np.ndarray
    ) -> tuple:
        """
        depth-first search on the adjacency matrix starting from the vertex v
        :param v: the vertex to start the depth-first search from
        :param marked: marked vertices
        :param unmarked: unmarked vertices
        :param A: Adjacency matrix of the graph
        :return: tuple
        """
        neighbors = {i for i, connected in enumerate(A[v]) if connected > 0}
        to_visit = neighbors.intersection(unmarked)

        for neighbor in to_visit:
            if neighbor in unmarked:
                marked.add(neighbor)
                unmarked.remove(neighbor)
                marked, unmarked = self._depth_first_search(
                    v=neighbor,
                    marked=marked,
                    unmarked=unmarked,
                    A=A
                )

        return marked, unmarked

    def is_graph_connected(
            self,
            adj: np.ndarray
    ) -> bool:
        """
        Checks if all nodes are reachable from node 0.
        :param adj: Adjacency matrix of the graph
        """
        if self.n_samples <= 1:
            return True

        start_node = 0
        marked = {start_node}
        unmarked = set(range(self.n_samples)) - {start_node}

        _, remaining = self._depth_first_search(
            v=start_node,
            marked=marked,
            unmarked=unmarked,
            A=adj
        )
        return len(remaining) == 0

    def get_adjacency(
            self,
            k: int,
            dist_subset: np.ndarray = None
    ):
        """
        builds a KNN adjacency matrix for a given k.
        :param k: number of nearest neighbors
        :param dist_subset: Optional distance matrix for a subset of points (used in recursion)
        """
        D = dist_subset if dist_subset is not None else self.dist_matrix
        n = len(D)

        if dist_subset is not None:
            indices = np.argsort(D, axis=0)
        else:
            indices = self.sorted_ind

        adj = np.zeros((n, n), dtype=int)
        for i in range(n):
            # indices[0] are self, so we take 1 to k+1
            nn = indices[1:k + 1, i]
            adj[i, nn] = 1
            adj[nn, i] = 1
        return adj

    def find_smallest_k(
            self,
            dist_subset: np.ndarray = None
    ) -> int:
        """
        Increments k until the graph becomes connected.
        :param dist_subset: Optional distance matrix
         for a subset of points (used in recursion)
        """
        k = self.min_k
        n = len(dist_subset) if dist_subset is not None else self.n_samples

        while k < n - 1:
            adj = self.get_adjacency(k=k, dist_subset=dist_subset)
            if self.is_graph_connected(adj=adj):
                return k
            k += 1
        return n - 1

    def find_components(self, adj: np.ndarray):
        """
        Identifies isolated islands in a disconnected graph.
        :param adj: Adjacency matrix of the graph
        """
        unmarked = set(range(len(adj)))
        components = np.zeros(len(adj), dtype=int)
        count = 0

        while unmarked:
            count += 1
            start_node = unmarked.pop()
            marked = {start_node}
            marked, unmarked = self._depth_first_search(
                v=start_node,
                marked=marked,
                unmarked=unmarked,
                A=adj
            )
            for node in marked:
                components[node] = count
        return components, count

    def build_refined_adj(
            self,
            dist_matrix: np.ndarray = None
    ) -> np.ndarray:
        """
        The recursive logic: ensures internal connectivity of clusters.
        :param dist_matrix: Optional distance matrix
        for a subset of points (used in recursion)
        """
        is_top_level = dist_matrix is None

        D = dist_matrix if dist_matrix is not None else self.dist_matrix
        k = self.find_smallest_k(dist_subset=D)

        # Save k only if we are at the top level
        if is_top_level:
            self.k = k

        adj = self.get_adjacency(k=k, dist_subset=D)

        # If we had to go above min_k, try to optimize the sub-islands
        if k > self.min_k:
            k_low = k - 1
            adj_low = self.get_adjacency(k=k_low, dist_subset=D)
            comps, n_comps = self.find_components(adj=adj_low)

            if n_comps > 1:
                for c in range(1, n_comps + 1):
                    indices = np.where(comps == c)[0]
                    if len(indices) > 1:
                        sub_dist = D[np.ix_(indices, indices)]
                        # Recursive call for the subcomponent
                        adj[np.ix_(indices, indices)] = self.build_refined_adj(dist_matrix=sub_dist)
        return adj

    def gaussian_kernel(self, sigma=None):
        if not sigma:
            knn_distances = self.true_dist_matrix[np.arange(self.n_samples), self.sorted_ind[self.k - 1]]
            self.sigma = np.median(knn_distances)
        else:
            self.sigma = sigma
        dist_sq = self.true_dist_matrix ** 2
        return np.exp(-dist_sq / (2 * (self.sigma ** 2)))

    def inverse_sq_euclidean_kernel(self):
        return 1.0 / (1.0 + self.true_dist_matrix ** 2)

    def compute_W(self, sigma=None):
        A = self.build_refined_adj()
        if self.kernel == 'gaussian':
            kernel_matrix = self.gaussian_kernel(sigma=sigma)
        elif self.kernel == 'inverse_sq_euclidean_d':
            kernel_matrix = self.inverse_sq_euclidean_kernel()
        else:
            raise ValueError(f"Unsupported kernel type: '{self.kernel}'. "
                             f"Must be 'gaussian' or 'inverse_sq_euclidean_d'.")
        W = np.where(A > 0, np.maximum(kernel_matrix, 1e-6), 0.0)
        return W
