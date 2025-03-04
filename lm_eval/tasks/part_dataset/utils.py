import pandas as pd
import datasets

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        choices = []
        for i in range(10):
            choice = doc.get(f'ending{i}','') 
            choices.append(choice if choice is not None else "No choice is available")
        out_doc = {
            "query": doc['question'],
            "choices": choices,
            "gold": int(doc["answer"]) -1,
        }
        return out_doc

    return dataset.map(_process_doc)