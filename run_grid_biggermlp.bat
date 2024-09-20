set lambda_s_values=0.1 0.2 0.5 1
set lambda_v_values=1 2 5 0.5
set lambda_pg_values=0.01 0.02 0.005
set policy_lr_values=0.1 0.05

for %%s in (%lambda_s_values%) do (
    for %%v in (%lambda_v_values%) do (
        for %%p in (%policy_lr_values%) do (
            echo Running with lambda_s=%%s, lambda_v=%%v, policy_lr=%%p
            python mlp_mnist_condg_out4.py --lambda_s %%s --lambda_v %%v --policy-lr %%p
        )
    )
)
