import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from omegaconf import ListConfig


class AuxiliaryMLP(pl.LightningModule):
    def __init__(self, hidden_dim=64, latent_dim=2, log_var_initializer=0.01, dropout_rate=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        
        if isinstance(log_var_initializer, (float, int, torch.Tensor)) and not hasattr(log_var_initializer, '__len__'):
            # Scalar case: fill with the same value
            var_tensor = torch.full((latent_dim,), log_var_initializer)
        elif isinstance(log_var_initializer, (ListConfig, list, tuple, np.ndarray, torch.Tensor)):
            # Vector case: convert to tensor and validate shape
            var_tensor = torch.tensor(log_var_initializer, dtype=torch.float32)
            if var_tensor.shape[0] != latent_dim:
                raise ValueError(f"Length of var_init ({var_tensor.shape[0]}) must match latent_dim ({latent_dim})")
        else:
            raise TypeError("var_init must be a float, int, list, tuple, or tensor")        
        
        # Inverse Softplus
        var_tensor = torch.log(torch.exp(var_tensor) - 1)        
        
        # Small weight initialization
        small_init = nn.init.normal_

        # Define layers
        self.dense1 = nn.Linear(latent_dim, hidden_dim)
        self.dropout1 = nn.Dropout(p=self.dropout_rate)  # 🔹 Dropout layer
        
        self.dense2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(p=self.dropout_rate)  # 🔹 Dropout layer
        
        self.dense3 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout3 = nn.Dropout(p=self.dropout_rate)  # 🔹 Dropout layer
        
        self.dense4 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout4 = nn.Dropout(p=self.dropout_rate)  # 🔹 Dropout layer
        
        self.dense8 = nn.Linear(hidden_dim, 2 * latent_dim)

        # Initialize weights manually
        small_init(self.dense1.weight, mean=0.0, std=0.01)
        small_init(self.dense8.weight, mean=0.0, std=0.01)

        # Bias initialization: [0.0]*latent_dim + [var_init]*latent_dim
        bias_init = torch.cat([
            torch.zeros(latent_dim),
            var_tensor
        ])
        with torch.no_grad():
            self.dense8.bias.copy_(bias_init)

    def forward(self, z):
        # z: (batch_size, latent_dim)
        z_aug = torch.cat([z, torch.zeros_like(z[:, :self.latent_dim])], dim=1)  # (batch_size, 2*latent_dim)

        x = F.relu(self.dense1(z))
        x = self.dropout1(x)
        
        # x = F.relu(self.dense2(x))
        # x = self.dropout2(x)
        
        # x = F.relu(self.dense3(x))
        # x = self.dropout3(x)

        # x = F.relu(self.dense4(x))
        # x = self.dropout4(x)
        
        out = self.dense8(x) + z_aug
        return out
    

class AuxiliaryMLPV2(pl.LightningModule):
    def __init__(self, hidden_dim=64, feature_dim=2048, output_dim=2, similarity_dim=50, init_alpha_sim=1, init_alpha_cls=1,
                 log_var_initializer=0.01, dropout_rate=0.1, fixed_var=False, linearly_combine_pca=False):
        super().__init__()
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.similarity_dim = similarity_dim
        self.dropout_rate = dropout_rate
        self.fixed_var = fixed_var
        self.linearly_combine_pca = linearly_combine_pca
        self.init_alpha_sim = init_alpha_sim
        self.init_alpha_cls = init_alpha_cls
        
        
        if isinstance(log_var_initializer, (float, int, torch.Tensor)) and not hasattr(log_var_initializer, '__len__'):
            # Scalar case: fill with the same value
            # var_tensor = torch.full((similarity_dim,), log_var_initializer)
            var_tensor = torch.full((1,), log_var_initializer)
        elif isinstance(log_var_initializer, (ListConfig, list, tuple, np.ndarray, torch.Tensor)):
            # Vector case: convert to tensor and validate shape
            var_tensor = torch.tensor(log_var_initializer, dtype=torch.float32)
            if var_tensor.shape[0] != similarity_dim:
                raise ValueError(f"Length of var_init ({var_tensor.shape[0]}) must match output_dim ({similarity_dim})")
        else:
            raise TypeError("var_init must be a float, int, list, tuple, or tensor")        
        
        # Inverse Softplus
        var_tensor = torch.log(torch.exp(var_tensor) - 1)        
        
        # Small weight initialization
        small_init = nn.init.normal_

        # Define layers
        self.dense1 = nn.Linear(feature_dim, hidden_dim)
        self.dropout1 = nn.Dropout(p=self.dropout_rate)  # Dropout layer         
        # self.dense2 = nn.Linear(hidden_dim, hidden_dim)
        # self.dropout2 = nn.Dropout(p=self.dropout_rate)  # Dropout layer  
        # self.dense3 = nn.Linear(hidden_dim, hidden_dim)
        # self.dropout3 = nn.Dropout(p=self.dropout_rate)  # Dropout layer   
        
        # self.dense4 = nn.Linear(hidden_dim, hidden_dim)
        # self.dropout4 = nn.Dropout(p=self.dropout_rate)  # Dropout layer 
        # self.dense5 = nn.Linear(hidden_dim, hidden_dim)
        # self.dropout5 = nn.Dropout(p=self.dropout_rate)  # Dropout layer 
        # self.dense6 = nn.Linear(hidden_dim, hidden_dim)
        # self.dropout6 = nn.Dropout(p=self.dropout_rate)  # Dropout layer                           
        
        if self.fixed_var:
            self.similarity_head = nn.Linear(hidden_dim, similarity_dim) #hidden_dim            
        else:
            #self.similarity_head = nn.Linear(hidden_dim, 2 * similarity_dim) #hidden_dim
            self.similarity_head = nn.Linear(hidden_dim, 1 + similarity_dim) #hidden_dim            
            #self.similarity_head = nn.Linear(hidden_dim, 1) #hidden_dim            
        self.classifcation_head = nn.Linear(hidden_dim, output_dim) #hidden_dim

        # Initialize weights manually
        small_init(self.dense1.weight, mean=0.0, std=0.01)
        small_init(self.similarity_head.weight, mean=0.0, std=0.01)
        small_init(self.classifcation_head.weight, mean=0.0, std=0.01)

        # Bias initialization: [0.0]*similarity_dim + [var_init]*similarity_dim
        if not self.fixed_var:                   
            bias_init = torch.cat([
                torch.zeros(similarity_dim),
                var_tensor
            ]) #var_tensor
         
            with torch.no_grad():
                self.similarity_head.bias.copy_(bias_init)
                #self.classifcation_head.bias.copy_(bias_init)
            
        if self.linearly_combine_pca:
            
            # FOR TISSUE AND CIFAR10 ALSO A SINGLE PARAMETER WORKS FINE                
            if self.output_dim <= 10:            
                self.alpha_sim = nn.Parameter(torch.tensor(self.init_alpha_sim, dtype=torch.float32))  #nn.Parameter(torch.randn(self.similarity_dim) * 0.01 + self.init_alpha_sim) ##  # # # 
                self.beta_sim = nn.Parameter(torch.empty(1).normal_(mean=0.0, std=0.01))  #nn.Parameter(torch.randn(self.similarity_dim) * 0.01) ## # # #
                self.alpha_cls = nn.Parameter(torch.tensor(self.init_alpha_cls, dtype=torch.float32)) #nn.Parameter(torch.randn(self.output_dim) * 0.01 + self.init_alpha_cls)  ### # #  
                self.beta_cls = nn.Parameter(torch.empty(1).normal_(mean=0.0, std=0.01)) #nn.Parameter(torch.randn(self.output_dim) * 0.01)  # #
                
            # FOR CIFAR100 FULL VECTOR REQUIRED.
            else:
                self.alpha_sim = nn.Parameter(
                    torch.full((self.similarity_dim,), float(self.init_alpha_sim), dtype=torch.float32)
                )
                self.beta_sim = nn.Parameter(
                    torch.empty(self.similarity_dim, dtype=torch.float32).normal_(mean=0.0, std=0.01)
                )

                self.alpha_cls = nn.Parameter(
                    torch.full((self.output_dim,), float(self.init_alpha_cls), dtype=torch.float32)
                )
                self.beta_cls = nn.Parameter(
                    torch.empty(self.output_dim, dtype=torch.float32).normal_(mean=0.0, std=0.01)
                )
            
    def forward(self, feats, logits, pca):
        # z: (batch_size, latent_dim)
        # z_aug = torch.cat([z, torch.zeros_like(z[:, :self.output_dim])], dim=1)  # (batch_size, 2*latent_dim)
        # logits_aug = torch.cat([logits, torch.zeros_like(logits[:, :self.output_dim])], dim=1)  # (batch_size, 2*latent_dim)
        #pca_aug = torch.cat([pca, torch.zeros_like(pca[:, :self.similarity_dim])], dim=1)  # (batch_size, 2*latent_dim)
        pca_aug = torch.cat([pca, torch.zeros(pca.size(0), 1, device=pca.device, dtype=pca.dtype)], dim=1)
        #print(self.alpha_cls.shape, self.beta_cls.unsqueeze(0).shape, logits.shape)
        
        x = F.relu(self.dense1(feats))
        x = self.dropout1(x)
        # x = F.relu(self.dense2(x))
        # x = self.dropout2(x)
        # x = F.relu(self.dense3(x))
        # x = self.dropout3(x)  
        
        # x = F.relu(self.dense4(x))
        # x = self.dropout4(x)  
        # x = F.relu(self.dense5(x))
        # x = self.dropout5(x)  
        # x = F.relu(self.dense6(x))
        # x = self.dropout6(x)        
                            
        if self.fixed_var:
            if self.linearly_combine_pca:
                similarity_out =  self.similarity_head(x) + self.alpha_sim*pca + self.beta_sim  # (batch_size, 2*similarity_dim)
                classification_out = self.classifcation_head(x) + self.alpha_cls*logits + self.beta_cls.unsqueeze(0)
            else:
                similarity_out = self.similarity_head(x) + pca #self.similarity_head(x) + self.alpha*pca + self.beta  # (batch_size, 2*similarity_dim)
                classification_out = self.classifcation_head(x) + logits # self.dense8(x) + z_aug
        else:
            similarity_out = self.similarity_head(x) + pca_aug  # (batch_size, 2*similarity_dim)
            classification_out = self.classifcation_head(x) + logits # self.dense8(x) + z_aug
            #similarity_out = self.similarity_head(x) + pca  # (batch_size, 2*similarity_dim)
            #similarity_out = torch.cat([pca, self.similarity_head(x)], dim=1) # + pca_aug  # (batch_size, 2*similarity_dim)
        
        #classification_out = self.classifcation_head(x) + logits # self.dense8(x) + z_aug
        
        return classification_out, similarity_out

"""       
class AuxiliaryMLPV2(pl.LightningModule):
    def __init__(self, hidden_dim=64, feature_dim=2048, output_dim=2, similarity_dim=50, init_alpha_sim=1, init_alpha_cls=1,
                 log_var_initializer=0.01, dropout_rate=0.1, fixed_var=False, linearly_combine_pca=False):
    # def __init__(self, hidden_dim=64, feature_dim=2048, output_dim=2, similarity_dim=50, 
    #              log_var_initializer=0.01, dropout_rate=0.1, linearly_combine_pca=False):
        super().__init__()
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.similarity_dim = similarity_dim
        self.dropout_rate = dropout_rate        
        self.linearly_combine_pca = linearly_combine_pca
        
        if isinstance(log_var_initializer, (float, int, torch.Tensor)) and not hasattr(log_var_initializer, '__len__'):
            # Scalar case: fill with the same value            
            var_tensor = torch.full((1,), log_var_initializer)
        else:
            raise TypeError("var_init must be a float, int, or tensor")        
        
        # Inverse Softplus
        var_tensor = torch.log(torch.exp(var_tensor) - 1)        
        
        # Small weight initialization
        small_init = nn.init.normal_

        # Define layers
        self.dense1 = nn.Linear(feature_dim, hidden_dim)
        self.dropout1 = nn.Dropout(p=self.dropout_rate)  # Dropout layer                                    
        
        self.similarity_head = nn.Linear(hidden_dim, similarity_dim)            
        self.classifcation_head = nn.Linear(hidden_dim, output_dim) 

        # Initialize weights manually
        small_init(self.dense1.weight, mean=0.0, std=0.01)
        small_init(self.similarity_head.weight, mean=0.0, std=0.01)
        small_init(self.classifcation_head.weight, mean=0.0, std=0.01)
            
        if self.linearly_combine_pca:   
            self.alpha_sim = nn.Parameter(torch.tensor(1., dtype=torch.float32)) 
            self.beta_sim = nn.Parameter(torch.empty(1).normal_(mean=0.0, std=0.01)) 
            self.alpha_cls = nn.Parameter(torch.tensor(1., dtype=torch.float32)) 
            self.beta_cls = nn.Parameter(torch.empty(1).normal_(mean=0.0, std=0.01)) 
            
    def forward(self, feats, logits, pca):        
        x = F.relu(self.dense1(feats))
        x = self.dropout1(x)

        if self.linearly_combine_pca:
            similarity_out =  self.similarity_head(x) + self.alpha_sim*pca + self.beta_sim 
            classification_out = self.classifcation_head(x) + self.alpha_cls*logits + self.beta_cls.unsqueeze(0)
        else:
            similarity_out = self.similarity_head(x) + pca 
            classification_out = self.classifcation_head(x) + logits 
        
        return classification_out, similarity_out    
"""    
    
    