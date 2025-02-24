import os

#return local path for local task
def get_data_path(file_name):

    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.realpath(__file__))

    # Build the relative path from the script's location
    data_path = os.path.join(current_dir, file_name)
    print(data_path)
    return data_path
