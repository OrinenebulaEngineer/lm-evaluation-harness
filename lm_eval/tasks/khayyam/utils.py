import pandas as pd
import datasets

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        out_doc = {
            "query": doc['Question Body'],
            "choices": [doc['Choice 1'], doc['Choice 2'], doc['Choice 3'], doc['Choice 4']],
            "gold": int(doc["Key"]) -1,
        }
        return out_doc

    return dataset.map(_process_doc)