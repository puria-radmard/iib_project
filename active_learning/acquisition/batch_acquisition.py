from .utils import *
from ..batch_querying import SequentialKMeansBatchQuerying


class VAEEnsembleKnowledgeUncertaintySubmodular(BatchAcquisition):

    def __init__(self, dataset, diversity_weighting):
        raise NotImplementedError
        self.diversity_weighting = diversity_weighting
        super(VAEEnsembleKnowledgeUncertaintySubmodular, self).__init__(dataset)

    def select_next_subset(self, candidate_windows, batchsize):
        raise NotImplementedError
        zs = [torch.mean(self.dataset.vae_ensemble_embeddings[w.i], dim = 0) for w in candidate_windows]
        distances = self.diversity_weighting * F.pdist(zs, 2)
        knowledge_uncertainties = [
            self.dataset.vae_ensemble_embeddings.instance_entropy(w.i) - 
                self.dataset.vae_ensemble_embeddings.instance_entropy(w.i)
            for w in candidate_windows
        ]


class SoftmaxKMeansDiscriminator(BatchAcquisition):
    # FINISH WITH WINDOWS IMPLEMENTATION
    """K-means the labelled (LIST OF) embeddings, and probabilistically select the furthest unlabelled sets 
    (min distance from a centroid)"""
    
    def __init__(self, dataset, embedding_feature, num_clusters):
        super(SoftmaxKMeansDiscriminator, self).__init__(dataset)
        self.embedding_feature = embedding_feature
        self.kmeans = KMeans(n_clusters=num_clusters, max_iter=10)
    
    def get_labelled_embeddings(self):
        embedding_attr = self.dataset.__getattr__(self.embedding_feature)
        labelled_seqs = [
            embedding_attr[i] for i in range(len(self.dataset))
            if self.dataset.index.has_any_labels(i)
        ]
        return [np.mean(ls) for ls in labelled_seqs]
        
    def get_unlabelled_embeddings(self):
        embedding_attr = self.dataset.__getattr__(self.embedding_feature)
        unlabelled_seqs = [
            embedding_attr[i] for i in range(len(self.dataset))
            if self.dataset.index.is_partially_unlabelled(i)
        ]
        return [np.mean(ls) for ls in unlabelled_seqs]   
    
    def generate_cluster_centers(self):
        unlabelled_embeddings = self.get_unlabelled_embeddings()
        self.kmeans.fit(all_points)
        return self.kmeans.cluster_centers_
    
    def select_next_subset(self, candidate_windows, batchsize):
        raise NotImplementedError("window implementation + replace with GMM")
        

class TFIDFFeatureFunctionBatchAcquisition(BatchAcquisition):
    """
    SUBMODULAR SUBSET SELECTION FOR LARGE-SCALE SPEECH TRAINING DATA - Wei et al 2014
        --> set tfidf_feature to trigrams - remember to make them jsonable
    This is a very basic, one step coreset selection. Can offer base for other, sequential variants.
    This class DOES NOT currently support subinstance annotation - only works for unit data/sequences.
    """

    def __init__(
        self,
        dataset,
        tfidf_weights,
        window_score_weight,
        #d_cache_path="d_cache.json",
        #index_log_path="submodular_log.log",
    ):
        super(TFIDFFeatureFunctionBatchAcquisition, self).__init__(dataset)
        self.current_indices = []
        self.tfidf_weights = tfidf_weights
        self.tfidf_attributes =  {k: self.dataset.__getattr__(k) for k in tfidf_weights.keys()}
        self.idf_caches = {k: {} for k in self.tfidf_weights.keys()}
        self.existing_mappers = {k: {} for k in self.tfidf_weights.keys()}
        self.window_score_weight = window_score_weight
        assert sum(tfidf_weights.values()) + self.window_score_weight, (f"Weights sum to {sum(tfidf_weights.values()) + window_score_weight}")

        print('Please ensure that TFIDFFeatureFunctionBatchAcquisition is not fed any overlapping windows')

        #self.idf_cache_path = d_cache_path
        #self.index_log_path = index_log_path

        # with open(d_cache_path, "r") as jfile:
        #     self.idf_cache = json.load(jfile)

        # with open(index_log_path, "r") as f:
        #     lines = f.read().split("\n")[:-1]
        #     if len(lines) > 0:
        #         all_indices = list(dataset.data.attr)
        #         self.load_previous_set([int(ind) for ind in lines], all_indices)

    def idf(self, feature, all_windows, component):
        """NEEDS TO BE TRIGGERED IF DATASET IS EXPANDED, MAINLY FOR ROUNDWISE ACQUISITION"""
        if feature in self.idf_caches[component].keys():
            return self.idf_caches[component][feature]
        else:
            d = 0
            for window in all_windows:
                ith_part_features = self.tfidf_attributes[component].get_attr_by_window(window)
                for utt_feautes in ith_part_features:
                    if feature in utt_feautes:
                        d += 1
                        break
            idf = np.log(self.V / d)
            self.idf_caches[component][feature] = idf
            return idf

    @staticmethod
    def g(x):
        return x ** 0.5

    @staticmethod
    def tf(feature, all_features):
        return all_features.count(feature)

    def m_u(self, feature, all_features, all_windows, component):
        # remember all_features is a list of list of features
        joined_features = []
        for af in all_features:
            joined_features.extend(af)
        return self.tf(feature, joined_features) * self.idf(feature, all_windows, component)

    @staticmethod
    def add_dictionaries(d1, d2):
        result = {key: d1.get(key, 0) + d2.get(key, 0) for key in set(d1) | set(d2)}
        return result

    def get_mu_dict(self, window, all_windows, component):
        part_features = self.tfidf_attributes[component].get_attr_by_window(window)
        unique_features = set()
        mu_dict = {}
        for utt_features in part_features:
            unique_features = unique_features.union(set(utt_features))
        for fe in unique_features:
            mu_dict[fe] = self.m_u(fe, part_features, all_windows, component)
        return mu_dict

    def f_feature(self, new_windows, all_windows, current_mapper, component):
        mu_all = current_mapper.copy()
        for window in new_windows:
            mu_j = self.get_mu_dict(window, all_windows, component)
            mu_all = self.add_dictionaries(mu_j, mu_all)
        return sum([self.g(m) for m in mu_all.values()]), mu_all

    def greedy_increment(self, candidate_windows, current_score, current_mappers):
        current_max_gain = 0
        chosen_idx = None
        next_mappers = None
        for i, w in enumerate(candidate_windows):    
            if i in self.current_indices:
                continue
            ith_score = w.score * self.window_score_weight
            candidate_mappers = {}
            for component in self.tfidf_weights.keys():
                component_ith_score, component_candidate_mapper = self.f_feature(
                    [w], candidate_windows, current_mappers[component], component
                )
                candidate_mappers[component] = component_candidate_mapper
                ith_score += component_ith_score * self.tfidf_weights[component]
            ith_gain = (ith_score - current_score) / sum(self.dataset.cost.get_attr_by_window(w))
            if ith_gain > current_max_gain:
                current_max_gain, chosen_idx, next_mappers = ith_gain, i, candidate_mappers
        next_score = current_score + current_max_gain
        # print(next_score, '\t', self.identification_feature.get_attr_by_window(chosen_window))
        return next_score, chosen_idx, next_mappers

    def select_next_subset(self, candidate_windows, total_cost):
        self.V = len(candidate_windows)
        score_history = []
        # self.idf_caches = {k: {} for k in self.tfidf_weights.keys()}
        cost = 0
        new_score = 0
        new_mu_mappers = {k: {} for k in self.tfidf_weights.keys()}
        for _ in tqdm(range(len(candidate_windows)), disable=not TQDM_MODE):
            new_score, new_idx, new_mu_mappers = self.greedy_increment(candidate_windows, new_score, new_mu_mappers)
            if new_idx == None:
                break
            self.current_indices.append(new_idx)
            new_window = candidate_windows[new_idx]
            score_history.append(new_score)
            # Typing here?
            cost += sum(self.dataset.cost.get_attr_by_window(new_window))
            if cost > total_cost:
                break
        return [candidate_windows[i] for i in self.current_indices]


class ScoreAugmentedTFIDFFeatureFunctionBatchAcquisition(
    TFIDFFeatureFunctionBatchAcquisition
):
    def __init__(
        self,
        dataset,
        tfidf_feature,
        score_attribute,
        d_cache_path="d_cache.json",
        index_log_path="submodular_log.log",
    ):
        super(ScoreAugmentedTFIDFFeatureFunctionBatchAcquisition, self).__init__(
            dataset, tfidf_feature, d_cache_path, index_log_path
        )
        self.score_attribute = self.dataset.__getattr__(score_attribute)

    def get_mu_dict(self, window, all_windows):
        all_features = self.tfidf_attribute.get_attr_by_window(window)
        score = self.score_attribute.get_attr_by_window(window)
        score = np.mean(score)
        unique_features = []
        mu_dict = {}
        for fe in all_features:
            if fe not in unique_features:
                unique_features.append(fe)
        for fe in unique_features:
            mu_dict[fe] = self.m_u(fe, all_features, all_windows) * float(score)
        return mu_dict


class KMeansCentroidBatchAcquisition(BatchAcquisition):
    """
    This is BADGE: https://arxiv.org/abs/1906.03671 if you set the relevant attribute to be made as:

        output = model(torch.tensor(batch).to('cuda'), False)
        hyp_preds = nn.functional.one_hot(output['last_logits'].argmax(axis = -1), 10)
        hyp_loss = (hyp_preds * output['last_logits']).sum()
        hyp_loss.backward()
        model.state_dict(keep_vars=True)['fc.weight'].grad.shape

    This would also require batch_size = 1 on the agent! This needs to be dealt with elsewhere
    """

    def __init__(self, dataset, attribute_name, pca_comps=16):
        super(KMeansCentroidBatchAcquisition, self).__init__(dataset)
        self.attribute_name = attribute_name
        self.pca_comps = pca_comps

    def select_next_subset(self, candidate_windows, batchsize):
        mechanism = SequentialKMeansBatchQuerying(
            batchsize, self.attribute_name, pca_comps=self.pca_comps
        )
        chosen_indices = mechanism.init_round(candidate_windows, self.dataset)
        return [DimensionlessAnnotationUnit(i, ..., None) for i in chosen_indices]
