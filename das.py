import os

# send a message to the console
l_batch_size  = [10, 50, 50, 50, 100, 100, 100, 200, 200]
l_accum_steps = [10,  1,  4, 10,   1,   2,   5,   1,   2]

l_allow_tf32 = [0, 1]
l_benchmark  = [0, 1]
l_precision  = ['16', '16-mixed', 'bf16', '32']
l_matmul_precision = ['medium', 'high']

for ii in range(len(l_batch_size)):
    batch_size = l_batch_size[ii]
    accum_steps = l_accum_steps[ii]
    for allow_tf32 in l_allow_tf32:
        for benchmark in l_benchmark:
            for precision in l_precision:
                for matmul_precision in l_matmul_precision:
                    print(f"batch_size: {batch_size}, accum_steps: {accum_steps}, allow_tf32: {allow_tf32}, benchmark: {benchmark}, precision: {precision}, matmul_precision: {matmul_precision}")
                    os.system(f"python pylghtng_dk.py --BATCH_SIZE {batch_size} --accum-step {accum_steps} --allow_tf32 {allow_tf32} --benchmark {benchmark} --precision {precision} --matmul_precision {matmul_precision}")