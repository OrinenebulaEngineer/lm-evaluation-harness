import pandas as pd
from datasets import Dataset

def process_data(dataset):
    # dataset = Dataset.from_pandas(dataframe)

    def _process_data(doc):
        # Ensure doc is a DataFrame before further processing
        # if not isinstance(doc, pd.DataFrame):
        #     doc = pd.DataFrame(doc)
        doc['key'] = int(doc['Key'])


        
        # if doc['Key'] == 1.0:
        #     doc['Key'] = 'Choice 1'
        # elif doc['Key'] == 2.0:
        #     doc['Key'] = 'Choice 2'
        # elif doc['Key'] == 3.0:
        #     doc['Key'] = 'Choice 3'
        # elif doc['Key'] == 4.0:
        #     doc['Key'] = 'Choice 4'

        # Create the output dictionary for each document
        out_doc = {
            "Question": doc['Question Body'],
            'Choice 1': doc['Choice 1'],
            'Choice 2': doc['Choice 2'],
            'Choice 3': doc['Choice 3'],
            'Choice 4': doc['Choice 4'],
            'Answer': doc['Key']
        }

        return out_doc
    
    # Apply the function to the dataframe
    return dataset.map(_process_data)

# Assuming `dataframe` is your pandas DataFrame
# processed_data = process_data(dataframe)
