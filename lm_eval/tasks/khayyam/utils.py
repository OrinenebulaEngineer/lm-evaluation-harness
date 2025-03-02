import pandas as pd
import datasets

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        out_doc = {
            "query": doc['Question Body'],
            "choices": ['Choice 1', 'Choice 2', 'Choice 3', 'Choice 4'],
            "gold": int(doc["Key"]) -1,
        }
        return out_doc

    return dataset.map(_process_doc)