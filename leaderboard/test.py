import os 

def change_directory_run(task_dir_name:str):

    current_directory = os.getcwd()
    while not current_directory.endswith('lm_evaluation_harness'):
        current_directory = os.path.dirname(current_directory)
        if current_directory == os.path.dirname(current_directory):
            print("Project root 'lm_harness' not found.")
            return
        
    target_path = os.path.join(current_directory, 'lm_eval', 'tasks', task_dir_name )
    try:
        os.chdir(target_path)
    except Exception as e:
        print(f"cant find task directory {e}")

    return target_path



