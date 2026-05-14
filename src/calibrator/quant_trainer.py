import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from utils.utils import *
from calibrator.quantisation_head import *

 
class VQClassifier(pl.LightningModule):
    """
    LightningModule that:
      features (B,2048) -> reshape (B,S,d) -> VQ -> quantized -> classifier -> logits
    """
    def __init__(
        self,
        kwargs,
        num_classes,
        feature_dim,
        feature_loader,
        backbone):
       
        super().__init__()
        self.save_hyperparameters(ignore=["kwargs"])
        
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.S = kwargs.models.S
        self.d = kwargs.models.d
        self.K = kwargs.models.K
        self.vq_decay = kwargs.models.vq_decay
        self.vq_eps = kwargs.models.vq_eps
        self.hidden = kwargs.models.hidden
        self.dropout = kwargs.models.dropout     
        self.L1 = kwargs.models.L1           
        self.backbone = backbone

        self.optimizer_cfg = kwargs.models.optimizer

        # Modules
        if self.backbone == 'vit':
            print("\n\n\nQUANTIZING A ViT ARCHITECTURE \n\n\n")
            self.bn = nn.BatchNorm1d(self.feature_dim)
        self.vq = VQHeadEMA(K=self.K, d=self.d, L1=self.L1, decay=self.vq_decay, eps=self.vq_eps)
        self.init_codebook(feature_loader)
        self.cls = QuantizedClassifierHead(S=self.S, d=self.d, num_classes=self.num_classes, hidden=self.hidden, dropout=self.dropout)        

    def forward(self, feats: torch.Tensor, return_entropy: bool = False):        
        # feats: (B, 2048) or already (B,S,d)
        if feats.dim() == 2:
            B, D = feats.shape
            assert D == self.feature_dim, f"Expected feature_dim={self.feature_dim}, got {D}"
            # if self.backbone == 'vit':
            #     feats = self.bn(feats)     
            z = feats.view(B, self.S, self.d)
        elif feats.dim() == 3:
            # z = feats
            B, S, d = feats.shape
            feats = feats.view(B, S * d)
            # if self.backbone == 'vit':
            #     feats = self.bn(feats)
            z = feats.view(B, S, d)
        else:
            raise ValueError(f"Unexpected feature shape: {tuple(feats.shape)}")

        if return_entropy:
            z_q, indices, entropy = self.vq(z, return_entropy=True)          # (B,S,d), (B,S)                    # (B,C)
            return z_q, indices, entropy
        else:
            z_q, indices = self.vq(z)                               # (B,S,d), (B,S)
            logits = self.cls(z_q)             # (B,C)            
            return logits, indices

    def training_step(self, batch, batch_idx=None):
        feats, _, _, y_one_hot, _, _ = batch     
        y = torch.argmax(y_one_hot, dim=1)
        
        logits, indices = self(feats)

        loss = F.cross_entropy(logits, y)        

        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()

        usage = self.vq.codeword_usage(indices)
        used_codewords = (usage > 0).float().sum()

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_used_codewords", used_codewords, on_step=False, on_epoch=True, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx=None):
        feats, _, _, y_one_hot, _, _ = batch     
        y = torch.argmax(y_one_hot, dim=1)
        
        logits, indices = self(feats)

        loss = F.cross_entropy(logits, y)

        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()

        usage = self.vq.codeword_usage(indices)
        used_codewords = (usage > 0).float().sum()

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_used_codewords", used_codewords, on_step=False, on_epoch=True, prog_bar=False)

    def configure_optimizers(self):
        # mirror your dynamic optimizer selection style
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
        optimizer = optimizer_class(self.cls.parameters(), **optimizer_kwargs)
        return optimizer

    @torch.no_grad()
    def init_codebook(self, feature_loader, max_batches: int = 50):
        """
        Initializes the codebook from real slot vectors pulled from your dataset.
        Call once before training (or at the start of fit).
        """
        samples = []
        for i, batch in enumerate(feature_loader):
            feats = batch[0]
            if feats.dim() == 2:
                B, D = feats.shape
                z = feats.view(B, self.S, self.d)
            else:
                z = feats
            samples.append(z.reshape(-1, self.d))
            if i + 1 >= max_batches:
                break

        samples = torch.cat(samples, dim=0).to(self.device)  # (Nslots, d)
        # if self.backbone == 'vit':
        #     self.vq.init_codebook_from_samples_std(samples)
        # else:
        self.vq.init_codebook_from_samples(samples)

    @torch.no_grad()
    def extract(self, batch):
        feats, _, _, y_one_hot, _, _ = batch         
        
        if feats.dim() == 2:
            B, D = feats.shape
            assert D == self.feature_dim, f"Expected feature_dim={self.feature_dim}, got {D}"
            z = feats.view(B, self.S, self.d)
        elif feats.dim() == 3:
            z = feats
        else:
            raise ValueError(f"Unexpected feature shape: {tuple(feats.shape)}")

        z_q, indices = self.vq(z)          # (B,S,d), (B,S)                
        logits = self.cls(z_q)             # (B,C)          
        preds = logits.argmax(dim=1).view(-1,1)      
        
        out = {
                "features": z_q.view(B, -1),               # (B, S*d), # quantized features   
                "logits": logits,
                "preds": preds,     
                "true": torch.argmax(y_one_hot, dim=-1).view(-1,1),
                "indices": indices
                }
        return out
    
    