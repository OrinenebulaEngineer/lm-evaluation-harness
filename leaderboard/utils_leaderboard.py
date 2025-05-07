import json
import pandas as pd
import os
from typing import Union

# import gradio as gr

def read_jsonl_file(jsonl_file_path:str):
    result_list = []
    if not os.path.exists(jsonl_file_path):
        print("file not found")
        return ["File not found", "", "", "", "", ""]
    with open(path, "r") as f:
        for line in f:
            try:
                 data = json.loads(line)

                 hellaswag = data.get("hellaswag", 0)
                 khayyam_challenge = data.get("khayyam-challenge", 0)

                 average_acc = (int(hellaswag) + int(khayyam_challenge))/2

                 result_list.append([   
                 data.get("Model", ""),
                 average_acc,
                 data.get("Precision", ""),
                 data.get("#Params (B)", ""),
                 hellaswag,
                 khayyam_challenge])
            except Exception as e:
                print([f"Error handling file {e}", "","", "","", ""])

            df = pd.DataFrame(result_list, columns=["Model","Average_Accuracy", "Precision", "#Params (B)", "hellaswag", "khayyam-challenge"])
            df_sorted = df.sort_values(by="Average_Accuracy", ascending=False)
    return df_sorted.values.tolist()

def lunch_show_results():
    headers = ["Model","Average_Accuracy", "Precision", "#Params (B)", "hellaswag", "khayyam-challenge"]
    with gr.Blocks() as app:
            gr.Markdown("# Results Viewer")  # title
            tabel = gr.Dataframe(
                 headers = headers,
                 datatype= "str",
                 value = read_jsonl_file(),
                 interactive=False       # not editable
            )

    app.launch()


import os 

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
            
            task_name :    acc,  
            "Model" :        model_name,
            "#Params (B)" :  model_params,
            "Precision" :    model_precision,
            "# Count result" : 0
        }

    if os.path.exists(jsonl_path):
        with open(jsonl_path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("Model") == model_name and entry[0]==task_name:
                    output["# Count result"] = output["# Count result"] + 1
    

    with open(jsonl_path,'a') as f:
        f.write(json.dumps(output) + "\n")

    return output




# path = "results.jsonl"

# df = read_jsonl_file(path)

# print(df)

