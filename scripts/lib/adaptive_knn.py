from scipy.spatial.distance import pdist, squareform
import numpy as np


class AdaptiveKNNGraph:
    def __init__(
            self,
            data: np.ndarray,
            min_k: int = 5,
            max_k: int = 15,
            edges_to_inject: list = [],
            kernel='gaussian'
    ):
        self.data = data
        self.min_k = min_k
        self.max_k = max_k
        self.dist_matrix = squareform(pdist(data, metric='euclidean'))
        self.n_samples = len(self.dist_matrix)
        self.kernel = kernel
        self.inject = edges_to_inject is not None and len(edges_to_inject) > 0
        if self.inject:
            self.true_dist_matrix = self.dist_matrix.copy()
            self.inject_random_edges(edges_to_inject)
        else:
            self.true_dist_matrix = self.dist_matrix
        self.sorted_ind = np.argsort(self.dist_matrix, axis=0)
        self.k = None
        self.sigma = None

    def inject_random_edges(
            self,
            pairs
    ):
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
        if len(adj) <= 1:
            return True

        start_node = 0
        marked = {start_node}
        unmarked = set(range(len(adj))) - {start_node}

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
        Increments k until the graph becomes connected, or until hitting the limit.
        :param dist_subset: Optional distance matrix
         for a subset of points (used in recursion)
        """
        k = self.min_k
        n = len(dist_subset) if dist_subset is not None else self.n_samples
        k_limit = n - 1 if self.inject else min(self.max_k, n - 1)

        while k < k_limit:
            adj = self.get_adjacency(k=k, dist_subset=dist_subset)
            if self.is_graph_connected(adj=adj):
                return k
            k += 1
        return k_limit

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

    def closest_pairs(
            self,
            comps: np.ndarray,
            n_comps: int,
            D: np.ndarray
    ) -> tuple:
        """
        Finds the minimum distance between every pair of components and the vertices realizing it.
        :param comps: component label of every vertex
        :param n_comps: number of connected components
        :param D: distance matrix of the graph
        :return: tuple
        """
        members = [np.where(comps == c)[0] for c in range(1, n_comps + 1)]
        weights = np.zeros((n_comps, n_comps))
        pairs = {}

        for a in range(n_comps):
            for b in range(a + 1, n_comps):
                sub_dist = D[np.ix_(members[a], members[b])]
                i, j = np.unravel_index(np.argmin(sub_dist), sub_dist.shape)
                weights[a, b] = weights[b, a] = sub_dist[i, j]
                pairs[(a, b)] = (members[a][i], members[b][j])
        return weights, pairs

    def minimum_spanning_edges(
            self,
            weights: np.ndarray
    ) -> list:
        """
        Prim's algorithm on a dense symmetric weight matrix.
        :param weights: pairwise weights between the vertices of the tree
        """
        n = len(weights)
        in_tree = np.zeros(n, dtype=bool)
        in_tree[0] = True
        best = weights[0].copy()
        source = np.zeros(n, dtype=int)
        edges = []

        for _ in range(n - 1):
            v = int(np.argmin(np.where(in_tree, np.inf, best)))
            edges.append((int(source[v]), v))
            in_tree[v] = True
            closer = weights[v] < best
            best[closer] = weights[v][closer]
            source[closer] = v
        return edges

    def connect_components(
            self,
            adj: np.ndarray,
            D: np.ndarray
    ) -> np.ndarray:
        """
        Joins the connected components along a minimum spanning tree over them.
        :param adj: Adjacency matrix of the graph
        :param D: distance matrix of the graph
        """
        comps, n_comps = self.find_components(adj=adj)
        if n_comps == 1:
            return adj

        weights, pairs = self.closest_pairs(comps=comps, n_comps=n_comps, D=D)
        for a, b in self.minimum_spanning_edges(weights=weights):
            vi, vj = pairs[(min(a, b), max(a, b))]
            adj[vi, vj] = 1
            adj[vj, vi] = 1
        return adj

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
        elif not self.k:
            raise ValueError("Global k is not defined on connected component level")
        elif k > self.k:
            raise ValueError("Connected component k {} exceeds global k {}".format(k, self.k))

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

        adj = self.connect_components(adj=adj, D=D)
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
