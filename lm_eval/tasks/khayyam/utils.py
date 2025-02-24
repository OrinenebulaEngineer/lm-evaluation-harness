import pandas as pd
from datasets import Dataset

def process_docs(dataset):
    # dataset = Dataset.from_pandas(dataframe)

 def _process_docs(doc):
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

    doc['Key'] = int(doc['Key'])

    # out_doc = {
    #     "Question": doc['Question Body'],
    #     'Key': int(doc['Key'])

    # }

    return doc

# Apply the function to each row of the DataFrame
    # Apply the function to the dataframe
 return dataset.map(_process_docs)

# Assuming `dataframe` is your pandas DataFrame
# processed_data = process_data(dataframe)
