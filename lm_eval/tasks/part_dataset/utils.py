import pandas as pd
import datasets

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        choices = [doc[f'ending{i}'] for i in range(10)]  # Assuming choices are labeled as 'Choice 1' to 'Choice 10'

        out_doc = {
            "query": doc['question'],
            "choices": choices,
            "gold": int(doc["answer"]) -1,
        }
        return out_doc

    return dataset.map(_process_doc)