import wandb
import random
import os
import time

os.environ['WANDB_SILENT']="true"
os.environ['WANDB_CONSOLE']='off'

# Vars
entity = 'importer-test'
project_name = '100_steps_20_metrics_200_runs_2MB_artifacts'
total_num_runs =  50
num_metrics_to_log = 20
num_steps_per_metric = 100
logged_artifact_size = 2 # False or int value for file size in MB

def generate_artifact(fileSize):
    file_path = 'artifact_3.file'
    with open(file_path, 'wb') as f:
        f.write(os.urandom(fileSize * 1000000))
    return file_path
start_time = time.time()
for i in range(total_num_runs):
    print(f"Starting run {i+1}/{total_num_runs}")

    wandb.init(
        entity = entity,
        project=project_name,

        # track hyperparameters and run metadata
        config={
        "learning_rate": 0.02,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": 10,
        }
    )

    # simulate training
    for step in range(num_steps_per_metric):
        log_dict = {}
        for metric_num in range(num_metrics_to_log):
            log_dict[f'metric_{metric_num}'] = random.randint(0,100)
        wandb.log(log_dict)

    # Log an Artifact
    if logged_artifact_size:
        path = generate_artifact(logged_artifact_size)
        art = wandb.Artifact(f"test_artifact_{wandb.run.id}", type="test")
        art.add_file(path)
        wandb.log_artifact(art)


    wandb.finish()
    elapsed_minutes = (time.time() - start_time)/60
    minutes_remaining  = round(((total_num_runs/(i+1)) * elapsed_minutes) - elapsed_minutes)
    print("Run Finished")
    print(f"{minutes_remaining} minutes remaining \n")
