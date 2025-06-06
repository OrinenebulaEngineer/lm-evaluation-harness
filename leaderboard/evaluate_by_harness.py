import sys
import logging
from datetime import datetime
import os
from manage_request import EvalRequest
from pathlib import Path


import json
from lm_eval import evaluator, utils
from manage_request import EvalRequest
from typing import Union
from config_logging import setup_logger
from lm_eval.tasks import TaskManager
import os
import utils_leaderboard 

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run_evaluatin(
        
        eval_request : EvalRequest,
        task_names : list,
        task_direcotry : str,
        num_fewshot: int,
        batch_size: Union[int, str],
        device: str,
        local_dir: str=None,
        result_repo: str=None,
        limit: int=None,
        model: str = "hf",
):
    """Runs one evaluation for the current evaluation request file, then pushes the results to the hub.
    Args:
        eval_request (EvalRequest): Input evaluation request file representation
        task_names (list): Tasks to launch
        num_fewshot (int): Number of few shots to use
        batch_size (int or str): Selected batch size or 'auto'
        device (str): "cpu" or "cuda:0", depending on what you assigned to the space
        local_dir (str): Where to save the results locally
        results_repo (str): To which repository to upload the results
        limit (int, optional): Whether to use a number of samples only for the evaluation - only for debugging
    Returns:
        _type_: _description_
    """
    if(task_direcotry == "khayyam-challenge"): #each custom task can add
        task_direcotry = f"lm_eval/tasks/{task_direcotry}"
        run_direcory = utils_leaderboard.change_directory(task_dir_name=task_direcotry)
        print(f"Current directory is {run_direcory}")

    logging.getLogger("openai").setLevel(logging.WARNING)
    logger = setup_logger(__name__)

    if limit:
        logger.info(
            "WARNING : -- limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT.")
        

    task_manager = TaskManager()
    
    all_tasks = task_manager.all_tasks
    
    task_names = utils.pattern_match(task_names,all_tasks)
    print(f"task name is {task_names}")
    results = evaluator.simple_evaluate(
                                    model = model,
                                    model_args = eval_request.get_model_args(),
                                    tasks = task_names,
                                    num_fewshot = num_fewshot,
                                    batch_size = batch_size,
                                    device = device,
                                    limit = limit,
                                    write_out=True,  # Whether to write out an example document and model input, for checking task integrity
                                 )

    results["config"]["model_dtype"] = eval_request.precision
    results["config"]["model_name"] = eval_request.model_arg
    results["config"]["model_sha"] = eval_request.revision

    dumped = json.dumps(results, indent=2)
    logger.info(dumped)

    results_path = Path(local_dir, eval_request.model_arg, f"results_{datetime.now()}.json")
    results_path.parent.mkdir(exist_ok=True, parents=True)
    results_path.write_text(dumped)
    results["config"]["results_path"] = results_path

    

    logger.info(utils.make_table(results))
    print(f"evaluation result saved to {results_path}")

    # Assuming this is part of your run_evaluatin function
    try:
        jsonl_dir = utils_leaderboard.change_directory("leaderboard")
        if jsonl_dir is None:
            raise ValueError("Failed to change to the 'leaderboard' directory.")
        
        print(f"jsonl_dir: {jsonl_dir}")  # Debugging output
        jsonl_file_path = os.path.join(jsonl_dir, "results.jsonl")
        print(f"jsonl_file_path: {jsonl_file_path}")  # Debugging output
    except Exception as e:
        logger.error(f"Error during directory change or file path creation: {e}")
        return  # Handle the error appropriately

    # if os.path.exists(jsonl_path):
    #     print("Directory already exists:", jsonl_path)
    # else:
    #     os.makedirs(jsonl_path, exist_ok=True)
    #     print("Model args directory created at:", jsonl_path)
    #     now = datetime.now()
    #     date_time_str = now.strftime("%Y%m%d_%H%M%S")
    #     result_path = f"result_{date_time_str}.jsonl"

    eval_result = utils_leaderboard.write_to_jsonl(results, jsonl_path=jsonl_file_path)
    return(eval_result,jsonl_file_path)



def read_json_file(file_path: str):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data 


def evaluate(model):
    eval_request = EvalRequest(
        model_arg=model, 
        json_filepath="results.jsonl",
        )
    tasks = ['hellaswag', 'khayyam-challenge', 'arc_easy'] #, 'khayyam-challenge', 'arc_easy'
    eval_results = []
    jsonl_path = ""
    for task in tasks:
        eval_result,jsonl_path= run_evaluatin(
            eval_request=eval_request,
            task_names=task,
            num_fewshot=0,
            batch_size="auto",
            task_direcotry=task,
            device='cuda',
            local_dir="output_result",
            limit = 5
            )
        print(eval_result)
        eval_results.append(eval_result)
    return eval_results

# eval_result = evaluate("google/gemma-2-9b")


