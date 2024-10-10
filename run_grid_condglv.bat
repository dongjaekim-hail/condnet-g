set lambda_s_values=5 10 15 20
set lambda_v_values=0.05 0.1 0.2 4
set learning_rate_values=0.05 0.01 0.1

for %%s in (%lambda_s_values%) do (
    for %%v in (%lambda_v_values%) do (
        for %%l in (%learning_rate_values%) do (
            echo Running with lambda_s=%%s, lambda_v=%%v, learning_rate=%%l
            python condg_mlp.py --lambda_s %%s --lambda_v %%v --learning_rate %%l
        )
    )
)
