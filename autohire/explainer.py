import numpy as np

from .model import BayesianMulticlassModel
from .bow import BagOfWords
from .encoder import LabelEncoder


class BayesianModelExplainer(BayesianMulticlassModel):
    """
    Explainer of the decision made by the base model
    """

    def __init__(self, label_encoder: LabelEncoder, bag_of_words: BagOfWords) -> None:
        super().__init__(len(label_encoder), len(bag_of_words))
        self.bag_of_words = bag_of_words
        self.label_encoder = label_encoder

    def explain(self, text=None, label_filter=None):
        """
        Visualize what are the prior probabilities of classes and which words
        add the the likelihood of each class.
        """
        class_frequencies = np.sum(self.counts, axis=1)
        word_frequencies = np.sum(self.counts, axis=0)

        prior = class_frequencies / np.sum(class_frequencies)  # p(label)
        likelihood = self.counts / np.expand_dims(
            class_frequencies, axis=1
        )  # p(word|label)
        evidence = word_frequencies / np.sum(word_frequencies)  # p(word)

        if text is not None:
            counts_vector = self.bag_of_words.get_counts(text)
            likelihood = np.multiply(likelihood, counts_vector)

        prior_ordering = np.flip(np.argsort(prior))
        for item in prior_ordering:
            likelihood = likelihood / (evidence + 0.00001)
            label = self.label_encoder.decode(item)
            word_ids = np.flip(np.argsort(likelihood[item]))
            word_ids = word_ids[:10]
            if label_filter is None or label in label_filter:
                print(f"{label}: {' '.join(self.bag_of_words.decode_data(word_ids))}")
