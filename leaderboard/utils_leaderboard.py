import json
import pandas as pd
import os
from typing import Union

# import gradio as gr

def read_jsonl_file(jsonl_file_path:str):
    result_list = []
    if not os.path.exists(jsonl_file_path):
        print("file not found")
        # return ["File not found", "", "", "", "", ""]
        return
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
    while not current_directory.endswith('lm_evaluation_harness'):
        current_directory = os.path.dirname(current_directory)
        if current_directory == os.path.dirname(current_directory):
            print("Project root 'lm_harness' not found.")
            return
        
    target_path = os.path.join(current_directory, task_dir_name )
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
    if task_name == 'khayyam-challenge':    
        output={
            
            "khayyam-challenge" : round((results['results'][task_name]["acc,none"])*100,2),
            "Model" : results["config"]["model_name"],
            "#Params (B)" : ((results["config"]["model_num_parameters"])//1000000000),
            "Precision" : results["config"]["model_dtype"]
        }
    

    with open(jsonl_path,'a') as f:
        f.write(json.dumps(output) + "\n")

    return output




# path = "results.jsonl"

# df = read_jsonl_file(path)

# print(df)

