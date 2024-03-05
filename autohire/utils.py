import re
from collections import defaultdict

import numpy as np
import pandas as pd
from pdfreader import PDFDocument, SimplePDFViewer, document

from .hyperparams import *


def clean_text(text: str):
    """
    Given text it removes all the non-character words, small words,
    converts everything to small letters, tokenizes and returns as a list.
    :param text: The text to be cleaned
    """
    text = text.lower()
    text = re.sub("[^a-z]", " ", text)
    data = text.split()
    data = list(filter(lambda x: len(x) >= WORD_LENGTH_THRESHOLD, data))
    return data


def parse_pdf(filename: str):
    """
    Read text from a PDF file.
    Clean the text, tokenize it, and return as a list of tokens.
    :param :
    """
    fd = open(filename, "rb")
    document = PDFDocument(fd)
    viewer = SimplePDFViewer(fd)
    output_strings = []
    for i in range(len(list(document.pages()))):
        viewer.navigate(1)
        viewer.render()
        output_strings.extend(viewer.canvas.strings)
    file_contents = " ".join(output_strings)
    return clean_text(file_contents)


def parse_resume_df():
    resume_df = pd.read_csv("data/resume-dataset.csv")
    resume_df["Keywords"] = resume_df["Resume"].apply(clean_text)
    return resume_df["Keywords"].values, resume_df["Category"].values
