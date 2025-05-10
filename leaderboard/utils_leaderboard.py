import json
import pandas as pd
import os
from typing import Union
import gradio as gr
from pathlib import Path
import shutil
from datasets import Dataset
from datasets import Dataset, Features, Value


# import gradio as gr

def jsonl_to_table(jsonl_file_path:str):
    records =[]
    if not os.path.exists(jsonl_file_path):
        print("file not found")
        return
    
    
    with open(jsonl_file_path, "r") as f:

        for line in f:
            try:
                 records.append(json.loads(line))
            except json.JSONDecodeError:
                 continue
            
            df = pd.DataFrame(records)


            pivot_df = df.pivot_table(index='Model', columns='Task', values='accuracy',  aggfunc='first').reset_index()
            extra_cols = df.groupby('Model')[["Precision", "#Params (B)"]].first()

            #Combine everything
            final_df = extra_cols.join(pivot_df).reset_index()

            return final_df.fillna("-")



def display_table():
    table = jsonl_to_table
    return gr.Dataframe(value=table, headers="keys", datatype="auto", label="Model Performance Table")





def lunch_show_results():
    demo = gr.Interface(fn = display_table, inputs=[], outputs= "dataframe", title="Model Accuracy Table" )
    return demo



def change_directory(task_dir_name:str):

    current_directory = os.getcwd()
    while not current_directory.endswith('lm-evaluation-harness'):
        current_directory = os.path.dirname(current_directory)
        if current_directory == os.path.dirname(current_directory):
            print("Project root 'lm-evaluation-harness' not found.")
            return
    print(f"this is current dit {current_directory}")    
    target_path = os.path.join(current_directory, task_dir_name )
    print(f"this is target path {target_path}")
    try:
        os.chdir(target_path)
    except Exception as e:
        print(f"cant find task directory {e}")

    return target_path

def upload_jsonl_to_hf(jsonl_path: str, repo_name: str, private: bool = False):
    """
    Upload a JSONL file as a Hugging Face dataset.
    Args:
        jsonl_path (str): Path to the .jsonl file.
        repo_name (str): The name of the dataset repository on Hugging Face, e.g. "username/result".
        private (bool): Whether the dataset should be private. Default is False (public).
    """
    # Load the .jsonl data
    with open(jsonl_path, "r") as f:
        data = [json.loads(line) for line in f]

    # Define the features explicitly
    features = Features({
        'Task': Value(dtype='string'),
        'accuracy': Value(dtype='float64'),
        'Model': Value(dtype='string'),
        '#Params (B)': Value(dtype='int64'),
        'Precision': Value(dtype='string'),
        '# Count result': Value(dtype='int64')
    })

    # Convert to Hugging Face Dataset with defined features
    dataset = Dataset.from_list(data, features=features)
    print(f"  ready for push to hf dataset")

    # Push to Hugging Face Hub
    dataset.push_to_hub(repo_name, private=private)
    print(f"   PUSH to hf dataset")






def write_to_jsonl(results, jsonl_path:str):

    # Check if 'results' and 'results[task_name]' exist
    if 'results' not in results or not results['results']:
        print("No results found.")
        return None

    keys= list(results['results'].keys())
    task_name = keys[0] 

    acc =  round((results['results'][task_name]["acc,none"])*100,2)
    model_name = results["config"]["model_name"]
    model_params=((results["config"]["model_num_parameters"])//1000000000)
    model_precision = results["config"]["model_dtype"]


    output={
            
            "Task" :    task_name,
            "accuracy" : acc,  
            "Model" :        model_name,
            "#Params (B)" :  model_params,
            "Precision" :    model_precision,
            "# Count result" : 0
        }

    filterd_line = []
    if os.path.exists(jsonl_path):
        with open(jsonl_path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                if  not (entry.get("Model") == model_name and entry.get("Task")==task_name):
                    filterd_line.append(line)
        
       
        with open(jsonl_path, 'w') as f:
            f.writelines(filterd_line)
    

    with open(jsonl_path,'a') as f:
        f.write(json.dumps(output) + "\n")

    #copy result.jsonl in hf_leaderboard
    current_dir = Path.cwd()

    # Define source and destination paths relative to current script
    source_path = current_dir / "results.jsonl"
    destination_path = current_dir.parent.parent / "HF_leaderboard" / "results.jsonl"
    # Make sure destination directory exists
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    # Copy the file
    shutil.copy(source_path, destination_path)
    print(f"Copied to {destination_path}")

    hf_dir_path = current_dir.parent.parent / "HF_leaderboard" 
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#last locatopn
    try:
        print(f"this is run directory{hf_dir_path}")
        os.chdir(hf_dir_path)
        result = upload_jsonl_to_hf("results.jsonl", "orinnebula/results")
    except Exception as e:
        print(f"cant find HF_leaderboard directory for push result dataset directory {e}")

    os.chdir(current_dir)
    return output






