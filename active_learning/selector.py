import os
import json
import logging
import torch
from .beam_search import GreedyBeamSearchSolution, EpsilonGreedyBeamSearchSolution # StochasticBeamSearchSolution
from .batch_querying import *
from .annotation_classes import SentenceSubsequenceAnnotationUnit, DimensionlessAnnotationUnit

TQDM_MODE = True


class Selector:
    def __init__(
        self,
        normalisation_index: float,
        round_cost,
        beam_search_parameter,
        acquisition,
        window_class,
        diversity_policy,
        selection_mode,
        epsilon=0.2
    ):
        self.normalisation_index = normalisation_index
        self.round_cost = round_cost
        self.round_selection = []
        self.all_round_windows = []
        self.beam_search_parameter = beam_search_parameter
        self.acquisition = acquisition
        self.window_class = window_class
        self.diversity_policy = diversity_policy

        self.epsilon = epsilon
        if selection_mode == 'argmax':
            self.beam_search_class = GreedyBeamSearchSolution
        elif selection_mode == 'epsilon_greedy':
            self.beam_search_class = EpsilonGreedyBeamSearchSolution
        elif selection_mode == 'softmax':
            self.beam_search_class = StochasticBeamSearchSolution
        else:
            raise ValueError(f'selection_mode "{selection_mode}" not valid')

    def window_generation(self, batch_indices, dataset):
        batch_unit_scores = self.acquisition.score(batch_indices)
        windows = []
        # TODO: TEST WITH SENTENCES AND WRITE ANY PROCESSING NEEDED HERE
        for i, unit_scores in zip(batch_indices, batch_unit_scores):
            window_args = self.score_extraction(unit_scores, i, dataset)
            windows.extend([
                self.window_class(i, window["bounds"], window["score"])
                for window in window_args
            ])
            # Do not filter out all labelled/partially labelled windows!
            # Needed for downstream logging
            [w.get_cost(dataset) for w in windows]
        return windows

    def assign_agent(self, agent):
        # REMOVE THIS AND DEPEND ON ACQUISITON DATASET
        self.agent = agent

    def score_extraction(self, unit_scores, i, dataset):
        raise NotImplementedError

    def score_aggregation(self, word_scores):
        raise NotImplementedError

    def initialise_solution(self):
        return self.beam_search_class(
            [],
            self.round_cost,
            self.beam_search_parameter,
            self.diversity_policy,
            None,
            None,
            {},
            self.epsilon
        )

    def initialise_solutions(self, unit_scores):
        # Initialise with best B scores
        b_solutions = [self.initialise_solution() for _ in range(self.beam_search_parameter)]
        b_solutions = [
            sol.add_window(unit_scores[j], self.agent.train_set)
            for j, sol in enumerate(b_solutions)
        ]
        return b_solutions

    def extend_solutions(self, b_solutions, unit_scores, usable_mask):
        temporary_solutions = []  # -> self.beam_search_parameter**2
        for solution in b_solutions:
            local_branch, usable_mask = solution.branch_out(
                temporary_solutions, unit_scores, usable_mask, train_set=self.agent.train_set
            )
            temporary_solutions.extend(local_branch)
        temporary_solutions.sort(key=lambda x: x.score, reverse=True)
        b_solutions = temporary_solutions[: self.beam_search_parameter]
        return b_solutions, usable_mask

    def select_best(self, window_scores):
        # window_scores = [(i, [r1, r2], score), ...]
        if TQDM_MODE:
            logging.info("beginning beam search: ")
            print("initialising diversity policy...")
        self.all_round_windows = window_scores
        # Filter out all labelled/partially labelled windows!
        window_scores =(
            list(filter(self.agent.train_set.index.new_window_unlabelled, window_scores))
        )
        self.diversity_policy.init_round(window_scores, self.agent.train_set)

        # i.e. the B solutions are completely disjoint! This might have to be changed later if beam search is
        # actually being used
        usable_mask = torch.tensor([1 for _ in window_scores])
        b_solutions = self.initialise_solutions(window_scores)
        usable_mask[:self.beam_search_parameter] = 0
        while all([not b.lock for b in b_solutions]):
            b_solutions, usable_mask = self.extend_solutions(b_solutions, window_scores, usable_mask)

        best_solution = max(b_solutions, key=lambda x: x.score)
        best_windows = best_solution.windows
        budget_spent = best_solution.cost

        self.round_selection = best_windows.copy()
        return best_windows, budget_spent

    def reduce_window_size(self):
        pass

    def save(self, save_path):
        # savable_lookup = [{"tokens": k, "labels": v} for k, v in self.labelled_ngrams.items()]
        with open(os.path.join(save_path, "round_selection.pk"), "w") as f:
            json.dump(
                {
                    "all_round_windows": [w.savable() for w in self.all_round_windows],
                    "round_selection_windows": [
                        w.savable() for w in self.round_selection
                    ],
                    # "cumulative_labelled_ngrams": savable_lookup
                },
                f,
            )

    @staticmethod
    def purify_entries(entries):
        """
        Sort and remove disjoint entries of form [([list, of, word, idx], score), ...]
        """
        start_entries = sorted(entries, key=lambda x: x[-1], reverse=True)
        final_entries = []
        highest_idx = set()
        for entry in start_entries:
            span = set(range(*entry[0]))
            if highest_idx.intersection(span):
                pass
            else:
                highest_idx = highest_idx.union(span)
                final_entries.append(entry)
        return final_entries


class DimensionlessSelector(Selector):
    def __init__(self, round_cost, acquisition, window_class, diversity_policy, selection_mode):
        super(DimensionlessSelector, self).__init__(
            normalisation_index=1,
            round_cost=round_cost,
            beam_search_parameter=1,
            acquisition=acquisition,
            window_class=window_class,
            diversity_policy=diversity_policy,
            selection_mode=selection_mode
        )

    def score_extraction(self, unit_scores, i, dataset):
        return [{"bounds": ..., "score": float(unit_scores.mean(dim=-1))}]


class FullSequenceSelector(Selector):
    def __init__(
        self,
        normalisation_index,
        round_cost,
        acquisition,
        diversity_policy,
        selection_mode
    ):
        window_class = SentenceSubsequenceAnnotationUnit
        super(FullSequenceSelector, self).__init__(
            normalisation_index=normalisation_index,
            round_cost=round_cost,
            beam_search_parameter=1,
            acquisition=acquisition,
            window_class=window_class,
            diversity_policy=diversity_policy,
            selection_mode=selection_mode
        )

    def score_aggregation(self, word_scores):
        """
        Standard score aggregation where word-wise scores are added or averaged
        """
        score = torch.sum(word_scores)
        score *= len(word_scores) ** (-self.normalisation_index)
        return score

    def score_extraction(self, unit_scores, i, dataset):
        """
        Input:
            scores_list: [list, of, scores, from, a, sentence, None, None]
            None REPRESENTS PREVIOUSLY LABELLED WORD - WHICH WILL NOT APPEAR FOR THIS STRATEGY
        Output:
            entries = [([list, of, word, idx], score), ...] for all possible extraction batches
            For this strategy, entries is one element, with all the indices of this sentence
        """

        score = self.score_aggregation(unit_scores)
        # This can be merged with dimensionless selector no?
        return [{"bounds": (0, len(unit_scores)), "score": score}]


class FixedSequenceWindowSelector(Selector):
    def __init__(
        self,
        window_size,
        beta,
        round_cost,
        beam_search_parameter,
        acquisition,
        diversity_policy,
        selection_mode
    ):
        window_class = SentenceSubsequenceAnnotationUnit
        super(FixedSequenceWindowSelector, self).__init__(
            normalisation_index=1.0,
            round_cost=round_cost,
            beam_search_parameter=beam_search_parameter,
            acquisition=acquisition,
            window_class=window_class,
            diversity_policy=diversity_policy,
            selection_mode=selection_mode
        )
        self.window_size = window_size
        self.beta = beta

    def score_aggregation(self, word_scores):
        """
        Standard score aggregation where word-wise scores are added or averaged
        """
        score = torch.sum(word_scores)
        score *= len(word_scores) ** (-self.normalisation_index)
        return score

    def reduce_window_size(self):
        self.window_size -= 1
        if self.window_size <= 0:
            self.window_size = 1

    def score_extraction(self, unit_scores, i, dataset):
        indices_and_word_scores = [([j, j + self.window_size], unit_scores[j : j + self.window_size])
                                   for j in range(len(unit_scores) - self.window_size + 1)]
        outlist = []
        for idx, scores in indices_and_word_scores:
            score = self.score_aggregation(scores)
            outlist.append({"bounds": idx, "score": score})
        return outlist


class VariableSequenceWindowSelector(Selector):
    def __init__(
        self,
        window_range,
        beta,
        round_cost,
        beam_search_parameter,
        normalisation_index,
        acquisition,
        diversity_policy
    ):
        window_class = SentenceSubsequenceAnnotationUnit
        super(VariableSequenceWindowSelector, self).__init__(
            normalisation_index=normalisation_index,
            round_cost=round_cost,
            beam_search_parameter=beam_search_parameter,
            acquisition=acquisition,
            window_class=window_class,
            diversity_policy=diversity_policy
        )
        self.window_range = window_range
        self.beta = beta

    def reduce_window_size(self):
        self.window_range[0] = min([1, self.window_range[0] - 1])
        self.window_range[1] = min([2, self.window_range[1] - 1])

    def score_extraction(self, unit_scores, i, dataset):
        """
        Input:
            scores_list: [list, of, scores, from, a, sentence, nan, nan]
            None REPRESENTS PREVIOUSLY LABELLED WORD
        Output:
            entries = [([list, of, word, idx], score), ...] for all possible extraction batches
        """
        indices_and_word_scores = []
        for w in range(*self.window_range):
            indices_and_word_scores.extend([([j, j+w], unit_scores[j:j+w]) for j in range(len(unit_scores)-w+1)])
        outlist = []
        for idx, scores in indices_and_word_scores:
            score = self.score_aggregation(scores)
            outlist.append({"bounds": idx, "score": score})
        return outlist


class SubsetSelector:
    def __init__(self, round_cost, acquisition, scorer, window_class=DimensionlessAnnotationUnit):
        # ADD PARENT
        self.round_cost = round_cost
        self.round_selection = []
        self.all_round_windows = []
        self.acquisition = acquisition  # ADD CHECK FOR THIS
        self.window_class = window_class
        self.scorer = scorer

    def score_aggregation(self, word_scores):
        score = torch.sum(word_scores)
        score /= len(word_scores)
        return score

    def score_extraction(self, unit_scores, i, dataset):
        score = self.score_aggregation(unit_scores)
        return [{"bounds": (0, len(unit_scores)), "score": score}]

    def window_generation(self, batch_indices, dataset):
        batch_unit_scores = self.acquisition.score(batch_indices)
        windows = []
        # TODO: TEST WITH SENTENCES AND WRITE ANY PROCESSING NEEDED HERE
        for i, unit_scores in zip(batch_indices, batch_unit_scores):
            window_args = self.score_extraction(unit_scores, i, dataset)
            windows.extend([
                self.window_class(i, window["bounds"], window["score"])
                for window in window_args
            ])
            # Do not filter out all labelled/partially labelled windows
            [w.get_cost(dataset) for w in windows]
        return windows

    def assign_agent(self, agent):
        self.agent = agent

    def select_best(self, window_scores):
        # Filter out all labelled/partially labelled windows!
        window_scores =(
            list(filter(self.agent.train_set.index.new_window_unlabelled, window_scores))
        )
        next_windows = self.acquisition.select_next_subset(window_scores, self.round_cost)
        return next_windows, len(next_windows)


class IndependentSubinstanceSubsetSelector(SubsetSelector):

    def __init__(self, round_cost, acquisition, scorer=None):
        window_class = SentenceSubsequenceAnnotationUnit
        super().__init__(round_cost, acquisition, scorer,  window_class)
    
    def window_generation(self, batch_indices, dataset):
        windows = []
        for i in batch_indices:
            windows.extend([self.window_class(i, [j, j+1], 0) for j in range(len(dataset.cost[i]))])
        [w.get_cost(dataset) for w in windows]
        return windows
