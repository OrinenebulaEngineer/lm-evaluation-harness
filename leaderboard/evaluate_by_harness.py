import sys
import logging

import os
from manage_request import EvalRequest


import json
from lm_eval import evaluator, utils
from manage_request import EvalRequest
from typing import Union
from config_logging import setup_logger
from lm_eval.tasks import TaskManager


def run_evaluatin(
        
        eval_request : EvalRequest,
        task_names : list,
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

    logging.getLogger("openai").setLevel(logging.WARNING)
    logger = setup_logger(__name__)

    if limit:
        logger.info(
            "WARNING : -- limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT.")
        

    task_manager = TaskManager()
    all_tasks = task_manager.all_tasks
    task_names = utils.pattern_match(task_names,all_tasks)

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
    results["config"]["model_name"] = eval_request.model
    results["config"]["model_sha"] = eval_request.revision

    dumped = json.dumps(results, indent=2)
    logger.info(dumped)

    logger.info(evaluator.make_table(results))
if __name__ == "__main__":
    eval_request = EvalRequest(
        model_arg="google/gemma-2-9b", 
        json_filepath="results.jsonl",
        )
    run_evaluatin(
        eval_request=eval_request,
        task_names=['khayyam_challenge'],
        num_fewshot=0,
        batch_size="auto",
        device='cuda',
        local_dir="output_result",
        )