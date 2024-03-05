import numpy as np

from autohire.utils import parse_pdf, parse_resume_df
from autohire.bow import BagOfWords
from autohire.encoder import LabelEncoder
from autohire.model import BayesianMulticlassModel
from autohire.explainer import BayesianModelExplainer


if __name__ == "__main__":
    x_train, y_train = parse_resume_df()
    bag_of_words = BagOfWords(x_train)
    label_encoder = LabelEncoder(y_train)

    x_train = bag_of_words.get_counts(x_train)
    y_train = label_encoder.encode(y_train)
    model = BayesianMulticlassModel(len(label_encoder), len(bag_of_words))
    model.fit(x_train=x_train, y_train=y_train)

    x_test_input = parse_pdf("data/resumes/computers_2.pdf")
    x_test = bag_of_words.get_counts(x_test_input)
    result = model.predict(x_train[0])
    result = label_encoder.decode(result)

    for job in result[:5]:
        print(job)

    explainable_model = BayesianModelExplainer(label_encoder, bag_of_words)
    explainable_model.fit(x_train=x_train, y_train=y_train)

    print(
        """
ANALYSIS OF TRAINED PRIOR
-------------------------"""
    )
    explainable_model.explain()

    print(
        """
ANALYSIS OF TRAINED EVIDENCE
----------------------------"""
    )
    explainable_model.explain(x_test_input)
