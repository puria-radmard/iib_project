from .utils import *


class TimeWeightedPhonemeicConfidence(UnitwiseAcquisition):
    def __init__(self, dataset):
        super().__init__(dataset=dataset)

    def score(self, batch_indices):
        batch_output_scores = []
        for i in batch_indices:
            cnfs = self.dataset.phoneme_confidence_scores[i]
            durs = self.dataset.phoneme_durations[i]
            output_scores = []
            for j, cnf in enumerate(cnfs):
                total_unc = - np.dot(cnf, durs[j]).item()
                total_dur = np.sum(durs[j])
                output_scores.append(total_unc/total_dur)
                if not len(output_scores):
                    batch_output_scores.append(torch.tensor([-1.0], dtype=torch.float))
                else:
                    batch_output_scores.append(torch.tensor(output_scores))
        return batch_output_scores


class TimeWeightedGraphemeicConfidence(UnitwiseAcquisition):
    def __init__(self, dataset):
        super().__init__(dataset=dataset)

    def score(self, batch_indices):
        batch_output_scores = []
        for i in batch_indices:
            cnfs = self.dataset.grapheme_confidence_scores[i]
            durs = self.dataset.grapheme_durations[i]
            output_scores = []
            for j, cnf in enumerate(cnfs):
                total_unc = - np.dot(cnf, durs[j]).item()
                total_dur = np.sum(durs[j])
                output_scores.append(total_unc/total_dur)
                if not len(output_scores):
                    batch_output_scores.append(torch.tensor([-1.0], dtype=torch.float))
                else:
                    batch_output_scores.append(torch.tensor(output_scores))
        return batch_output_scores


class TimeWeightedWordConfidence(UnitwiseAcquisition):
    def __init__(self, dataset):
        super().__init__(dataset=dataset)

    def score(self, batch_indices):
        batch_output_scores = []
        for i in batch_indices:
            cnfs = self.dataset.word_confidence_scores[i]
            durs = self.dataset.word_durations[i]
            output_scores = []
            for j, cnf in enumerate(cnfs):
                total_unc = - np.dot(cnf, durs[j]).item()
                total_dur = np.sum(durs[j])
                output_scores.append(total_unc/total_dur)
            if not len(output_scores):
                batch_output_scores.append(torch.tensor([-1.0], dtype=torch.float))
            else:
                batch_output_scores.append(torch.tensor(output_scores))
        return batch_output_scores


class AverageGraphemeicConfidence(UnitwiseAcquisition):
    def __init__(self, dataset):
        super().__init__(dataset=dataset)

    def score(self, batch_indices):
        batch_output_scores = []
        for i in batch_indices:
            cnfs = self.dataset.grapheme_confidence_scores[i]
            output_scores = []
            for j, cnf in enumerate(cnfs):
                output_scores.append(- np.mean(cnf))
                if not len(output_scores):
                    batch_output_scores.append(torch.tensor([-1.0], dtype=torch.float))
                else:
                    batch_output_scores.append(torch.tensor(output_scores))
        return batch_output_scores


class AverageGraphemeicConfidence(UnitwiseAcquisition):
    def __init__(self, dataset):
        super().__init__(dataset=dataset)

    def score(self, batch_indices):
        batch_output_scores = []
        for i in batch_indices:
            cnfs = self.dataset.phoneme_confidence_scores[i]
            output_scores = []
            for j, cnf in enumerate(cnfs):
                output_scores.append(- np.mean(cnf))
                if not len(output_scores):
                    batch_output_scores.append(torch.tensor([-1.0], dtype=torch.float))
                else:
                    batch_output_scores.append(torch.tensor(output_scores))
        return batch_output_scores


class AverageWordConfidence(UnitwiseAcquisition):
    def __init__(self, dataset):
        super().__init__(dataset=dataset)

    def score(self, batch_indices):
        batch_output_scores = []
        for i in batch_indices:
            cnfs = self.dataset.word_confidence_scores[i]
            output_scores = []
            for j, cnf in enumerate(cnfs):
                output_scores.append(- np.mean(cnf))
                if not len(output_scores):
                    batch_output_scores.append(torch.tensor([-1.0], dtype=torch.float))
                else:
                    batch_output_scores.append(torch.tensor(output_scores))
        return batch_output_scores


class BinaryClassifierPredictionAcquisition(UnitwiseAcquisition):
    """
        Preds are in [p(unlabelled), p(labelled)] => use pred[0] as acq score
        see daf_metrics.LabelledRankMetric
    """
    def __init__(self, dataset):
        super().__init__(dataset=dataset)

    def score(self, batch_indices):
        # Initialise return
        batch_output_scores = []

        # Iterate over instances (parts) in batch
        for i in batch_indices:

            # Get the two size vectors
            preds = self.dataset.classifier_preds[i]

            # Initialise the scores for this instance (1 per utt)
            output_scores = []

            for j, pred in enumerate(preds):
                
                # For each utt, append the p(unlabelled)
                output_scores.append(pred[0])

                # Sometimes empty, unlikely here though I think
                if not len(output_scores):
                    # Give lowest value if missing
                    batch_output_scores.append(torch.tensor([0.], dtype=torch.float))
                else:
                    batch_output_scores.append(torch.tensor(output_scores))
        return batch_output_scores


class TimeNormalisedBinaryClassifierPredictionAcquisition(UnitwiseAcquisition):
    """
        Exact same as above, except I divide by duration
        i.e. p(unlabelled)/T
    """
    def __init__(self, dataset):
        super().__init__(dataset=dataset)

    def score(self, batch_indices):
        # Initialise return
        batch_output_scores = []

        # Iterate over instances (parts) in batch
        for i in batch_indices:

            # Get the two size vectors and the durections
            preds = self.dataset.classifier_preds[i]
            durs = self.dataset.cost[i]

            # Initialise the scores for this instance (1 per utt)
            output_scores = []

            for j, (pred, dur) in enumerate(zip(preds, durs)):
                
                # For each utt, append the p(unlabelled)
                output_scores.append(pred[0] / dur)

                # Sometimes empty, unlikely here though I think
                if not len(output_scores):
                    # Give lowest value if missing
                    batch_output_scores.append(torch.tensor([0.], dtype=torch.float))
                else:
                    batch_output_scores.append(torch.tensor(output_scores))

        return batch_output_scores