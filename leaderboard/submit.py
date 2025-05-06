
import gradio as gr
from datasets import load_dataset, Dataset
from huggingface_hub import login
from datetime import datetime
import pytz
import uuid
import os
from dotenv import load_dotenv
import requests


from dotenv import load_dotenv
import os

from huggingface_hub import login



DATASET_NAME = "orinnebula/request"  # Change to your actual dataset

def submit_request(
    model_name, revision, precision, weight_type,
    model_type, params, license_str, private_bool
):
    try:
        # Load or create dataset
        try:
            dataset = load_dataset(DATASET_NAME, split="train")
        except Exception as e:
            dataset = Dataset.from_list([])

        # Tehran time
        tehran = pytz.timezone('Asia/Tehran')
        now = datetime.now(tehran)
        persian_time = now.isoformat(timespec='microseconds')

        new_entry = {
            "id": str(uuid.uuid4()),
            "model": model_name,
            "revision": revision,
            "precision": precision,
            "weight_type": weight_type,
            "submitted_time": persian_time,
            "model_type": model_type,
            "params": float(params),
            "license": license_str,
            "private": bool(private_bool)
        }

        dataset = dataset.add_item(new_entry)
        dataset.push_to_hub(DATASET_NAME)
        try:
            response = requests.post()

        return f"✅ Submitted! ID: {new_entry['id']}"

    except Exception as err:
        return f"❌ Error: {str(err)}"


iface = gr.Interface(
    fn=submit_request,
    inputs=[
        gr.Textbox(label="Model Name"),
        gr.Dropdown(["main"], label="Revision"),
        gr.Dropdown(["fp16", "bf16", "int8", "int4"], label="Precision"),
        gr.Dropdown(["Original"], label="Weight type"),
        gr.Dropdown(["⭕ : instruction-tuned", "🟢 : pretrained", "🔶 : fine-tuned"], label="Model Type"),
        gr.Number(label="Params (Billions)"),
        gr.Dropdown(["custom", "mit", "apache-2.0"], label="License"),
        gr.Checkbox(label="Private Model")
    ],
    outputs=gr.Textbox(label="Submission Status"),
    title="Submit Model Request",
    description="Fill the fields to submit model info to a Hugging Face dataset."
)

iface.launch(server_port=7861)



