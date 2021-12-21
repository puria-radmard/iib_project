import random

TQDM_MODE = True


class BeamSearchSolution:
    def __init__(
        self,
        windows,
        max_cost,
        B,
        diversity_policy,
        init_cost,
        init_score,
        init_overlap_index,
    ):
        self.windows = windows
        if not init_score:
            self.score = sum([w.score for w in windows])
        else:
            self.score = init_score
        if not init_cost:
            self.cost = sum([w.cost for w in windows])
        else:
            self.cost = init_cost
        self.overlap_index = init_overlap_index
        self.max_cost = max_cost
        self.lock = False
        self.B = B
        self.diversity_policy = diversity_policy

    def add_window(self, new_window, train_set):
        if self.cost >= self.max_cost:
            self.lock = True
            return self
        init_cost = self.cost + new_window.cost
        init_score = self.score + new_window.score
        init_overlap_index = self.overlap_index.copy()
        if new_window.i in init_overlap_index:
            init_overlap_index[new_window.i] = init_overlap_index[new_window.i].union(
                new_window.get_index_set()
            )  # Need to generalise this
        else:
            init_overlap_index[new_window.i] = new_window.get_index_set()
        # new_ngram = train_set.data_from_window(new_window)

        return self.__class__(
            self.windows + [new_window],
            self.max_cost,
            self.B,
            init_cost=init_cost,
            init_score=init_score,
            init_overlap_index=init_overlap_index,
            diversity_policy=self.diversity_policy,
        )

    def is_permutationally_distinct(self, other):
        # We do a proxy-check for permutation invariance by checking for score and cost of solutions
        if abs(self.score - other.score) < 1e-6 and self.cost == other.cost:
            return False
        else:
            return True

    def all_permutationally_distinct(self, others):
        for other_solution in others:
            if not self.is_permutationally_distinct(other_solution):
                return False
        else:
            return True

    def new_window_unlabelled(self, new_window):
        if new_window.i not in self.overlap_index:
            self.overlap_index[new_window.i] = set()  # Just in case!
            return True
        else:
            new_word_idx = new_window.get_index_set()
            if self.overlap_index[new_window.i].intersection(new_word_idx):
                return False
            else:
                return True

    def new_window_viable(self, new_window):
        if not self.new_window_unlabelled(new_window):
            return False
        if self.diversity_policy.new_window_viable(new_window, self):
            return True
        else:
            return False

    def branch_out(self, other_solutions, window_scores, usable_mask, train_set):
        raise NotImplementedError


class GreedyBeamSearchSolution(BeamSearchSolution):

    def __init__(
            self,
            windows,
            max_cost,
            B,
            diversity_policy,
            init_cost,
            init_score,
            init_overlap_index,
            *args
    ):
        super(GreedyBeamSearchSolution, self).__init__(
            windows,
            max_cost,
            B,
            diversity_policy,
            init_cost,
            init_score,
            init_overlap_index,
        )

    def ep_check(self):
        return False

    def choose_window_uniformly(self, window_scores, usable_mask, train_set):
        raise NotImplementedError

    def branch_out(self, other_solutions, window_scores, usable_mask, train_set):
        # ASSUME window_scores ALREADY SORTED
        local_branch = []
        for j, window in enumerate(window_scores):
            if not usable_mask[j]:
                continue
            if self.ep_check():
                possible_node, usable_mask = self.choose_window_uniformly(window_scores, usable_mask, train_set)
                local_branch.append(possible_node)
                if len(local_branch) == self.B:
                    return local_branch, usable_mask
            if self.new_window_viable(window):
                possible_node = self.add_window(window, train_set)
                # Permutation check unused if we are using usable_mask - this might want to be parameterised in the
                # future. You can use j here to change things up
                # if possible_node.all_permutationally_distinct(other_solutions):
                local_branch.append(possible_node)
                usable_mask[j] = 0
                if len(local_branch) == self.B:
                    return local_branch, usable_mask
            if self.lock:
                return [self], usable_mask

        # No more windows addable
        if len(local_branch) == 0:
            self.lock = True
            return [self], usable_mask
        else:
            return local_branch, usable_mask


class EpsilonGreedyBeamSearchSolution(GreedyBeamSearchSolution):

    def __init__(
            self,
            windows,
            max_cost,
            B,
            diversity_policy,
            init_cost,
            init_score,
            init_overlap_index,
            epsilon
    ):
        super(EpsilonGreedyBeamSearchSolution, self).__init__(
            windows,
            max_cost,
            B,
            diversity_policy,
            init_cost,
            init_score,
            init_overlap_index,
        )
        self.epsilon = epsilon

    def ep_check(self):
        return random.uniform(0, 1)<self.epsilon

    def choose_window_uniformly(self, window_scores, usable_mask, train_set):
        usable_indices = [i for i in range(len(usable_mask)) if usable_mask[i] == 1]
        random.shuffle(usable_indices)