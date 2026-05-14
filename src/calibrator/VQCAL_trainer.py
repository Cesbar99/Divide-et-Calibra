import pytorch_lightning as pl
from calibrator.VQDirCal import VQDirCal
import torch
import torch.nn.functional as F
import torch.optim as optim


class VQCalibrator(pl.LightningModule):
    def __init__(self, vqclassifier, kwargs): #K: int, C: int, lr: float = 1e-3, wd: float = 0.0, learn_pi: bool = False):
        super().__init__()
        #self.automatic_optimization = False
        
        self.vqclassifier = vqclassifier
        self.corruption_type = kwargs.corruption_type if hasattr(kwargs, "corruption_type") else None
        self.vqclassifier.eval()
        for p in self.vqclassifier.parameters():
            p.requires_grad = False
        self.diag = kwargs.models.diag
        self.learn_pi = kwargs.models.learn_pi     
        self.learn_bias = kwargs.models.learn_bias
        self.random = kwargs.models.random   
        self.standard_dirichlet = kwargs.models.standard_dirichlet
        self.quadratic = kwargs.models.quadratic

        self.cal = VQDirCal(K=kwargs.models.K, C=kwargs.dataset.num_classes, S=kwargs.models.S, diag=self.diag, quadratic=self.quadratic, learn_pi=self.learn_pi, learn_bias=self.learn_bias, random=self.random, standard_dirichlet=self.standard_dirichlet)
        self.optimizer_cfg = kwargs.models.optimizer
        self.lambda_reg = kwargs.models.lambda_reg
        self.quantization_only = kwargs.models.quantization_only
        
    def forward(self, feats: torch.Tensor):
        # feats: (B, 2048) or already (B,S,d)
        if feats.dim() == 2:
            B, D = feats.shape
            #assert D == self.feature_dim, f"Expected feature_dim={self.feature_dim}, got {D}"
            z = feats.view(B, self.vqclassifier.S, self.vqclassifier.d)
        elif feats.dim() == 3:
            z = feats
        else:
            raise ValueError(f"Unexpected feature shape: {tuple(feats.shape)}")

        z_q, indices = self.vqclassifier.vq(z)          # (B,S,d), (B,S)
        logits = self.vqclassifier.cls(z_q)             # (B,C)        
        p_hat = F.softmax(logits, dim=1)
        
        calibrated_probs, log_scores, alpha = self.cal(p_hat, indices)
        
        return calibrated_probs, log_scores, alpha, logits, indices

    def training_step(self, batch):
        feats, _, _, y_one_hot, _, _ = batch
        y = torch.argmax(y_one_hot, dim=1)
             
        with torch.no_grad():
            logits, indices = self.vqclassifier(feats) #logits, indices = self.vqclassifier(feats)          
            p_hat = F.softmax(logits, dim=1)

        calibrated_probs, log_scores, alpha = self.cal(p_hat, indices)                
        loss = F.cross_entropy(log_scores, y)                 

        self.log("cal_train_loss", loss, on_epoch=True, prog_bar=True)        
        self.log("cal_train_acc", (calibrated_probs.argmax(1) == y).float().mean(), on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch):
        feats, logits, _, y_one_hot, _, _ = batch
        y = torch.argmax(y_one_hot, dim=1)
                
        with torch.no_grad():            
            logits, indices = self.vqclassifier(feats)
            p_hat = F.softmax(logits, dim=1)

        calibrated_probs, log_scores, alpha = self.cal(p_hat, indices)                
        loss = F.cross_entropy(log_scores, y)         
        
        self.log("cal_val_loss", loss, on_epoch=True, prog_bar=True)        
        self.log("cal_val_acc", (calibrated_probs.argmax(1) == y).float().mean(), on_epoch=True, prog_bar=True)            

    def configure_optimizers(self):        
        opt_name = self.optimizer_cfg.name
        opt_name = opt_name[0].upper() + opt_name[1:]        
        if opt_name == "Adamw":
            opt_name = "AdamW"        
        lr = self.optimizer_cfg.lr
        wd = self.optimizer_cfg.get("weight_decay", 0.0)

        if "dam" in opt_name:
            optimizer_class = getattr(torch.optim, opt_name, None)
        elif "gd" in opt_name:
            optimizer_class = getattr(torch.optim, "SGD", None)
            opt_name = "SGD"
        else:
            optimizer_class = getattr(torch.optim, opt_name, None)

        if optimizer_class is None:
            raise ValueError(f"Unsupported optimizer: {opt_name}")

        optimizer_kwargs = {"lr": lr, "weight_decay": wd}
        if opt_name == "SGD":
            optimizer_kwargs["momentum"] = self.optimizer_cfg.get("momentum", 0.9)

        # IMPORTANT: only optimize classifier head; codebook is EMA-updated
        optimizer = optimizer_class(self.cal.parameters(), **optimizer_kwargs)
                
        return optimizer
                
    @torch.no_grad()
    def extract(self, batch):        
        feats, logits, init_pca, y_one_hot, _, _ = batch       
        #init_feats, init_logits, init_pca, y_one_hot, init_preds, init_preds_one_hot = batch  
        
        if feats.dim() == 2:
            B, D = feats.shape
            #assert D == self.feature_dim, f"Expected feature_dim={self.feature_dim}, got {D}"
            z = feats.view(B, self.vqclassifier.S, self.vqclassifier.d)
        elif feats.dim() == 3:
            z = feats
        else:
            raise ValueError(f"Unexpected feature shape: {tuple(feats.shape)}")

        z_q, indices, entropy = self.vqclassifier(z, return_entropy=True) # self.vqclassifier.vq(z, return_entropy=True)          # (B,S,d), (B,S)
        # z_flat = z.view(z.shape[0], -1)
        # z_q_flat = z_q.view(z_q.shape[0], -1)
        # diff = z_flat - z_q_flat
        # var = z_flat.var(dim=0, unbiased=False) + 1e-12
        l2 = entropy.mean(dim=1) #torch.norm(z_flat - z_q_flat, dim=1) #torch.sum((z_flat - z_q_flat) ** 2, dim=1) # Distance from centroids, (B,S)        
        logits = self.vqclassifier.cls(z_q)             # (B,C)        
        p_hat = F.softmax(logits, dim=1)
        
        calibrated_probs, log_scores, alpha = self.cal(p_hat, indices)
                
        if self.random:
            preds = log_scores.argmax(dim=1).view(-1,1)      
            out = {
                "features": z.view(B, -1),               # (B, S*d), # quantized features alternatively use original features  
                "logits": log_scores,
                "preds": preds,     
                "true": torch.argmax(y_one_hot, dim=-1).view(-1,1),
                "indices": indices,
                "alpha": alpha.view(B, -1)
                }
        elif self.quantization_only:     
            preds = logits.argmax(dim=1).view(-1,1)             
            out = {
                    "features": z_q.view(B, -1), #z_q.view(B, -1),               # (B, S*d), # quantized features alternatively use original features  
                    "logits": logits,
                    "preds": preds,     
                    "true": torch.argmax(y_one_hot, dim=-1).view(-1,1),
                    "indices": indices,
                    "alpha": alpha.view(B, -1),
                    "l2": l2
                    }
        elif self.corruption_type is not None:
            preds = log_scores.argmax(dim=1).view(-1,1)
            out = {
                "features": z_q.view(B, -1),
                # z_q.view(B, -1),               # (B, S*d), # quantized features alternatively use original features
                "logits": log_scores,
                "preds": preds,
                "true": torch.argmax(y_one_hot, dim=-1).view(-1, 1),
                "indices": indices,
                "alpha": alpha.view(B, -1),
                "pca": init_pca.view(B, -1),
                "l2": l2
            }
        else:
            preds = log_scores.argmax(dim=1).view(-1,1)      
            out = {
                    "features": z_q.view(B, -1), #z_q.view(B, -1),               # (B, S*d), # quantized features alternatively use original features  
                    "logits": log_scores,
                    "preds": preds,     
                    "true": torch.argmax(y_one_hot, dim=-1).view(-1,1),
                    "indices": indices,
                    "alpha": alpha.view(B, -1),
                    "l2": l2
                    }
        return out
    
    @torch.no_grad()
    def print_calibrator_params(self):
        print('A', self.cal.A_code.weight.shape)
        print(self.cal.A_code.weight)
        print('\n B', self.cal.B_code.weight.shape)
        print(self.cal.B_code.weight)
        
        

