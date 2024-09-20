set lambda_s_values=1 3 5 10
set lambda_v_values=0.1 0.2 0.5 1

for %%s in (%lambda_s_values%) do (
    for %%v in (%lambda_v_values%) do (
        echo Running with lambda_s=%%s, lambda_v=%%v
        python main_condg2.py --lambda_s %%s --lambda_v %%v
    )
)
