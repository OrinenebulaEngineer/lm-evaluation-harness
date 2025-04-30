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





path = "results.jsonl"

df = read_jsonl_file(path)

print(df)

