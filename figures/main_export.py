import pandas as pd
import wandb
import json

api = wandb.Api()
entity, project = "hails", "condgnet"
runs = api.runs(entity + "/" + project)

# 만약 tau=0.3을 뽑고 싶다면
target_tau = 0.3
c = 0
summary_list, config_list, name_list = [], [], []
for run in runs:
    # .summary contains the output keys/values
    #  for metrics such as accuracy.
    #  We call ._json_dict to omit large files
    res = json.loads(run.json_config)
    if res['tau']['value']==target_tau:
        c += 1


        summary_list.append(run.summary._json_dict)


        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k,v in run.config.items()
             if not k.startswith('_')})

        # .name is the human-readable name of the run.
        name_list.append(run.name)

runs_df = pd.DataFrame({
    "summary": summary_list,
    "config": config_list,
    "name": name_list
    })

runs_df.to_csv("project.csv")