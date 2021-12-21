from .data_utils import *

TQDM_MODE = True


class EarlyStopper:
    def __init__(self, model, patience: int):  # , maximise: bool):
        """
        An early stopping & callback class.
        patience is an integer, the number of epochs that a non-optimal statistic is allowed (adding number of steps soon)
        maximise is set to True for scores, False for losses
        """
        self.patience = patience
        # self.maximise = maximise
        self.model_state_dict = None
        self.model = model
        self.saved_epoch = 0
        self.scores = []
        self.min_score = float("inf")

    def is_overfitting(self, score):
        """
        Unused
        """
        scores = self.scores
        if len(scores) < self.patience:
            self.scores.append(score)
            return False

        if score < self.min_score:
            self.model_state_dict = self.model.state_dict()
            self.min_score = score

        scores.append(score)
        all_increasing = True
        s0 = scores[0]
        for s1 in scores[1:]:
            if s0 >= s1:
                all_increasing = False
                break
            s0 = s1
        self.scores = scores[1:]

        if all_increasing:
            print("reloading model\n")
            self.model.load_state_dict(self.model_state_dict)

        return all_increasing

    def check_stop(self, stats_list):

        if len(stats_list) > self.patience:
            if stats_list[-1] < stats_list[-2]:
                self.model_state_dict = self.model.state_dict()
                self.saved_epoch = len(stats_list)

        if self.patience < 0 or len(stats_list) < self.patience:
            return False
        if stats_list[-self.patience :] == sorted(stats_list[-self.patience :]):
            print(f"reloading from epoch {self.saved_epoch}")
            self.model.load_state_dict(self.model_state_dict)
            return True
        else:
            return False


class RenovationError(Exception):
    def __init__(self, message):
        # Call the base class constructor with the parameters it needs
        super().__init__(message)