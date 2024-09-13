# condnet-g
cnn  
optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-4)  
  
mlp  
optim.SGD(mlp_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)  
  
policy  
optim.Adam(gnn_policy.parameters(), lr=0.001, weight_decay=1e-4)