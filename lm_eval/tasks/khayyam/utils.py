import pandas as pd
from datasets import Dataset

def process_data(dataset):
    # dataset = Dataset.from_pandas(dataframe)

 def _process_data(doc):
    # Ensure that Key is treated as a float (to handle possible type differences)
    # key_value = float(doc['Key'])

    # if key_value == 1.0:
    #     doc['Key'] = 'Choice 1'
    # elif key_value == 2.0:
    #     doc['Key'] = 'Choice 2'
    # elif key_value == 3.0:
    #     doc['Key'] = 'Choice 3'
    # elif key_value == 4.0:
    #     doc['Key'] = 'Choice 4'

    out_doc = {
        "Question": doc['Question Body'],
        'Choice 1': doc['Choice 1'],
        'Choice 2': doc['Choice 2'],
        'Choice 3': doc['Choice 3'],
        'Choice 4': doc['Choice 4'],
        'gold': int(doc['Key'])
    }

    return out_doc

# Apply the function to each row of the DataFrame
    # Apply the function to the dataframe
    return dataset.map(_process_data)

# Assuming `dataframe` is your pandas DataFrame
# processed_data = process_data(dataframe)
