import pandas as pd
from datasets import Dataset

def process_data(dataframe : pd.DataFrame) :
    def _process_data(doc):
        dataset = Dataset.from_pandas(doc)

        if dataset['key'] == 1.0:
            dataset['key'] = 'Choice 1'
        elif dataset['key'] == 2.0:
            dataset['key'] = 'Choice 2'
        elif dataset['key'] == 3.0:
            dataset['key'] = 'Choice 3'
        elif dataset['key'] == 4.0:
            dataset['key'] = 'Choice 4'

        out_doc = {
            "Question" : dataset['Question Body'],
            'Choice 1' : dataset['Choice 1'],
            'Choice 2' : dataset['Choice 2'],
            'Choice 3' : dataset['Choice 3'],
            'Choice 4' : dataset['Choice 4'],
            'Answer'   : dataset['key']
        }

        return out_doc
    return dataframe.map(_process_data)


