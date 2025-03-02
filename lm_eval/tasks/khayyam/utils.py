import pandas as pd
from datasets import Dataset

def doc_to_target(dataset):
    # dataset = Dataset.from_pandas(dataframe)

 #def _process_docs(doc):
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


    # Ensure the 'Key' field is an integer and process it
    # Check if dataset is a DataFrame or a single document (dictionary)
    if isinstance(dataset, pd.DataFrame):
        # Apply the transformation to the 'Key' column if it's a DataFrame
        dataset['Key'] = dataset['Key'].apply(lambda x: int(x))
    else:
        # If it's a single dictionary, just convert the 'Key' field to int
        if isinstance(dataset, dict):
            dataset['gold'] = int(dataset.get('Key'))  # Default to 0 if 'Key' is missing


    # out_doc = {
    #     "Question": doc['Question Body'],
    #     'Key': int(doc['Key'])

    # }

  #  return doc

# Apply the function to each row of the DataFrame
    # Apply the function to the dataframe
#  return dataset.map(_process_docs)
    return dataset

# Assuming `dataframe` is your pandas DataFrame
# processed_data = process_data(dataframe)
