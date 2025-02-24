import pandas as pd
from datasets import Dataset

def process_data(dataframe : pd.DataFrame) :
    def _process_data(doc):
        dataset = Dataset.from_pandas(doc)

        if doc['key'] == 1.0:
            doc['key'] = 'Choice 1'
        elif doc['key'] == 2.0:
            doc['key'] = 'Choice 2'
        elif doc['key'] == 3.0:
            doc['key'] = 'Choice 3'
        elif doc['key'] == 4.0:
            doc['key'] = 'Choice 4'

        out_doc = {
            "Question" : doc['Question Body'],
            'Choice 1' : doc['Choice 1'],
            'Choice 2' : doc['Choice 2'],
            'Choice 3' : doc['Choice 3'],
            'Choice 4' : doc['Choice 4'],
            'Answer'   : doc['key']
        }

        return out_doc
    return dataframe.map(_process_data)


