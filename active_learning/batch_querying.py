# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# from sklearn.metrics import pairwise_distances_argmin_min
# from sklearn.neighbors import BallTree
import numpy as np

TQDM_MODE = True

class BatchQueryingBase:
    def __init__(self):
        pass

    def init_round(self, candidate_windows, train_set):
        raise NotImplementedError

    def new_window_viable(self, new_window, existing_solution):
        raise NotImplementedError


class NoPolicyBatchQuerying(BatchQueryingBase):
    def __init__(self):
        super(NoPolicyBatchQuerying, self).__init__()

    def init_round(self, candidate_windows, train_set):
        pass

    def new_window_viable(self, new_window, existing_solution):
        return True


class SequentialKMeansBatchQuerying(BatchQueryingBase):
    def __init__(self, num_clusters, attribute_name, pca_comps=8):
        super(SequentialKMeansBatchQuerying, self).__init__()
        self.kmeans = KMeans(n_clusters=num_clusters, max_iter=1, verbose=2)
        self.pca = PCA(pca_comps)
        self.attribute_name = attribute_name
        self.cluster_cache = None
        self.train_set = None

    def init_round(self, candidate_windows, train_set):
        train_set_attr = train_set.__getattr__(self.attribute_name)
        all_points = np.vstack(
            [train_set_attr.get_attr_by_window(w) for w in candidate_windows]
        )
        all_points = self.pca.fit_transform(all_points)
        self.cluster_cache = self.kmeans.fit_predict(all_points)
        # Fix to work with windows somehow!
        self.cluster_cache = {
            cw.i: self.cluster_cache[j] for j, cw in enumerate(candidate_windows)
        }
        self.train_set = train_set

        # For acquisition.KMeansCentroidBatchAcquisition
        closest, _ = pairwise_distances_argmin_min(
            self.kmeans.cluster_centers_, all_points
        )
        # Change this to windows when you have time
        return [w.i for j, w in enumerate(candidate_windows) if j in closest]

    def get_cluster(self, window):
        if not np.isnan(self.cluster_cache[window.i]):
            return self.cluster_cache[window.i][window.bounds]
        else:
            # TODO: So messy!!
            relevant_slice = self.train_set[window.i]
            relevant_slice.__getattr__(self.attribute_name)
            relevant_attr = np.array(relevant_slice[0].reshape(1, -1), dtype=np.float)
            self.cluster_cache[window.i] = self.kmeans.predict(relevant_attr)
            return self.cluster_cache[window.i][window.bounds]

    def new_window_viable(self, new_window, existing_solution):
        occupied_clusters = [self.get_cluster(w) for w in existing_solution.windows]
        # TODO: sort this out for structured data
        if self.get_cluster(new_window) in occupied_clusters:
            return False
        else:
            return True


class IncreasingAverageDistanceBatchQuerying(BatchQueryingBase):
    def __init__(self, attribute_name, pca_comps=8):
        super(IncreasingAverageDistanceBatchQuerying, self).__init__()
        self.pca = PCA(pca_comps)
        self.nearest_neighbours = None
        self.attribute_name = attribute_name
        self.train_set = None

    def init_round(self, candidate_windows, train_set):
        train_set_attr = train_set.__getattr__(self.attribute_name)
        all_points = np.vstack(
            [train_set_attr.get_attr_by_window(w) for w in candidate_windows]
        )
        self.pca.fit(all_points)
        self.train_set = train_set
        self.distance_history = []

    def new_window_viable(self, new_window, existing_solution):
        # TODO: sort this out for structured data
        train_set_attr = self.train_set.__getattr__(self.attribute_name)
        # Do we have to make this every time we want to check??
        existing_data = np.vstack(
            [train_set_attr.get_attr_by_window(w) for w in existing_solution.windows]
        )
        existing_data = self.pca.transform(existing_data)
        new_data = train_set_attr.get_attr_by_window(new_window).reshape(1, -1)
        new_data = self.pca.transform(new_data)
        tree = BallTree(existing_data, leaf_size=2)
        dist, _ = tree.query(new_data, k=1)
        dist = float(dist[0])
        average_dist = np.mean(self.distance_history)
        if len(self.distance_history) == 0 or dist > 0.8 * average_dist:
            self.distance_history.append(dist)
            return True
        else:
            return False


class MetadataBalanceBatchQuerying(BatchQueryingBase):
    def __init__(self, attribute_names, batch_cost):
        super(MetadataBalanceBatchQuerying, self).__init__()
        self.attribute_names = attribute_names
        self.target_proportion_dict = None
        self.current_proportion_dict = None
        self.batch_cost = batch_cost
        self.train_set = None

    def init_round(self, candidate_windows, train_set):
        target = {}
        current = {}
        for attribute_name in self.attribute_names:
            attribute = train_set.__getattr__(attribute_name)
            composition = [attribute.get_attr_by_window(w) for w in candidate_windows]
            values, counts = np.unique(composition)
            counts /= len(composition)
            counts *= self.batch_cost
            target[attribute_name] = {v: counts[j] for j, v in enumerate(values)}
            target[attribute_name] = {v: 0 for j, v in enumerate(values)}
        self.target_proportion_dict = target
        self.current_proportion_dict = current
        self.train_set = train_set

    def new_window_viable(self, new_window, existing_solution):
        for attribute_name in self.attribute_names:
            new_metadata = self.train_set.__getattr__(attribute_name).get_attr_by_window(new_window)
            if (
                self.current_proportion_dict[attribute_name][new_metadata] + new_window.cost >
                self.target_proportion_dict[attribute_name][new_metadata]
            ):
                return False
        for attribute_name in self.attribute_names:
            new_metadata = self.train_set.__getattr__(attribute_name).get_attr_by_window(new_window)
            self.current_proportion_dict[attribute_name][new_metadata] += new_window.cost
        return True