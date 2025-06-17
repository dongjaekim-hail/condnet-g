# CondGNet
MLP  
condg_mlp_scheduler.py 

CNN  
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
MLP  
mlp_mnist_cond_out.py 

CNN  
cnn_cifar10_cond_out.py

# LTH
MLP  
mlp_mnist_unstruc_reallth_tau.py -> unstructured    
mlp_mnist_struc_reallth_tau.py -> structured

CNN  
cnn_cifar10_unstruc_reallth_tau.py -> unstructured    
cnn_cifar10_struc_reallth_tau.py -> structured

# Runtime Magnitude Based Pruning
MLP  
mlp_mnist_runtime_weight_out.py -> weight based     
mlp_mnist_runtime_activation_out.py -> activation based

CNN  
cnn_cifar10_runtime_weight_out.py -> weight based      
cnn_cifar10_runtime_activation_out2.py -> activation based

# Plot
plot_barlegend.py -> bar graph legend   
plot_legend.py -> allacctau, cum legend  
plot_sub_allacctau.py -> all model acc, tau graph    
plot_sub_bar.py -> all model specificity bar graph  
plot_sub_cum.py -> all model GFLOPS graph   
plot_sub_lth.py -> unstructured, structured lth acc, tau graph (total pruning iterations)
