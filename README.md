# Branch
sjusju

# CondGNet
mlp  
condg_mlp_scheduler.py 

cnn  
condg_cnn_scheduler.py
  

  
| Hyper-parameter    | MLP     | CNN     |
|--------------------|---------|---------|
| λₛ                 | 7       | 7       |
| λᵥ                 | 0.2     | 0.5     |
| λ<sub>pg</sub>                | 0.001   | 0.001   |
| τ                  | 0.6     | 0.4     |
| Learning rate      | 0.03    | 0.0003  |
| GNN learning rate  | 0.03    | 0.001   |
| Warm up            | 0       | 0       |
| Multi              | 0.99    | 0.999   |
| Batch size         | 256     | 60      |
| Weight decay       | 0.0005  | 0.0005  |
| Epoch              | 200     | 200     |

# CondNet
mlp  
mlp_mnist_cond_out.py 

cnn  
cnn_cifar10_cond_out.py

# LTH
mlp  
mlp_mnist_unstruc_reallth_tau.py    
mlp_mnist_struc_reallth_tau.py

cnn  
cnn_cifar10_unstruc_reallth_tau.py    
cnn_cifar10_struc_reallth_tau.py

# Runtime Magnitude Based Pruning
mlp  
mlp_mnist_runtime_weight_out.py     
mlp_mnist_runtime_activation_out.py

cnn  
cnn_cifar10_runtime_weight_out.py     
cnn_cifar10_runtime_activation_out2.py