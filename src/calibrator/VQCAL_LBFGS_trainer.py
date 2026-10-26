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

        self.cal = VQDirCal(K=kwargs.models.K, C=kwargs.dataset.num_classes, S=kwargs.models.S, diag=self.diag, quadratic=self.quadratic, learn_pi=self.learn_pi, learn_bias=self.learn_bias, random=self.random, standard_dirichlet=self.standard_dirichlet, random_seed=kwargs.seed)
        self.optimizer_cfg = kwargs.models.optimizer

        self.lambda_a = kwargs.models.get("lambda_a", 0.0)
        if self.lambda_a is None:
            self.lambda_a = 0.0

        self.lambda_b = kwargs.models.get("lambda_b", 0.0)
        if self.lambda_b is None:
            self.lambda_b = 0.0

        self.lambda_sigma = kwargs.models.get("lambda_sigma", 0.0)
        if self.lambda_sigma is None:
            self.lambda_sigma = 0.0

        self.eps_reg = 1.0
        self.quantization_only = kwargs.models.quantization_only
        self.finetune_only = kwargs.models.finetune_only
        self.codeword_counts = None

    def set_codeword_counts(self, counts: torch.Tensor):
        """Set precomputed codeword utilization counts."""
        self.codeword_counts = counts

    def _compute_codeword_regularization_loss(self, indices: torch.Tensor, T: torch.Tensor):
        """
        Compute regularization on codeword factors with utilization weighting.

        Args:
            indices: VQ indices for current batch, shape (B, S)
            T: Diagonal factors, shape (S,)
            indices: VQ indices for current batch, shape (B, S)

        Returns:
            reg_loss: Total regularization loss
            weighted_terms: Dictionary of individual regularization terms
        """
        K = self.cal.K
        C = self.cal.C
        S = indices.shape[1]
        device = indices.device

        if self.codeword_counts is None:
            self.codeword_counts = torch.ones(K, device=device)

        n_c = self.codeword_counts.to(device)

        weight_c = K / (n_c + self.eps_reg)

        A = self.cal.A_code(indices.long()) #.weight
        B = self.cal.B_code(indices.long()) #.weight
        a_init = self.cal.initial_A_code[indices.long()].detach()
        b_init = self.cal.initial_B_code[indices.long()].detach()
        #a_init = #self.cal.A_code.weight.clone().detach()
        #b_init = #self.cal.B_code.weight.clone().detach()

        a_diff = A - a_init
        b_diff = B - b_init

        a_l2_sq = (a_diff ** 2).sum(dim=2)
        b_l2_sq = (b_diff ** 2).sum(dim=2)

        indices_flat = indices.view(-1)  # Flatten to (B*S,)
        
        a_l2_sq_flat = a_l2_sq.view(-1)  # Flatten to (B*S,)
        a_reg_per_cw = torch.zeros(K, device=device)
        a_reg_per_cw.scatter_add_(0, indices_flat, a_l2_sq_flat)
        a_reg = (weight_c * a_reg_per_cw).sum() / indices.numel()
        
        b_l2_sq_flat = b_l2_sq.view(-1)  # Flatten to (B*S,)
        b_reg_per_cw = torch.zeros(K, device=device)
        b_reg_per_cw.scatter_add_(0, indices_flat, b_l2_sq_flat)
        b_reg = (weight_c * b_reg_per_cw).sum() / indices.numel()

        t_reg = ((T ** 2 - 1.0) ** 2).sum()
        weight_r = 1.0 / (S + self.eps_reg)
        t_reg = weight_r * t_reg

        weighted_terms = {
            "a": self.lambda_a * a_reg,
            "b": self.lambda_b * b_reg,
            "sigma": self.lambda_sigma * t_reg,
        }

        reg_loss = sum(weighted_terms.values(), A.new_zeros(()))
        return reg_loss, weighted_terms
        
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
        if self.finetune_only:
            logits = self.vqclassifier.cls(z)             # (B,C)
        else:
            logits = self.vqclassifier.cls(z_q)             # (B,C)        
        p_hat = F.softmax(logits, dim=1)
        
        if self.diag:
            calibrated_probs, log_scores, alpha, indices, T = self.cal(p_hat, indices)
        else:
            calibrated_probs, log_scores, alpha, indices = self.cal(p_hat, indices)
        
        return calibrated_probs, log_scores, alpha, logits, indices

    @torch.no_grad()
    def _collect_full_dataset(self, data_loader, device):
        """Precompute frozen classifier outputs and concatenate every sample."""
        probabilities, all_indices, labels = [], [], []
        self.vqclassifier.eval()

        for batch in data_loader:
            feats, _, _, y_one_hot, _, _ = batch
            feats = feats.to(device)
            logits, indices = self.vqclassifier(feats)
            probabilities.append(F.softmax(logits, dim=1).cpu())
            all_indices.append(indices.cpu())
            labels.append(y_one_hot.argmax(dim=1).cpu())

        if not probabilities:
            raise ValueError("Cannot fit VQCalibrator on an empty data loader.")

        return (
            torch.cat(probabilities, dim=0).to(device),
            torch.cat(all_indices, dim=0).to(device),
            torch.cat(labels, dim=0).to(device),
        )

    def _full_batch_loss(self, probabilities, indices, labels, regularize):
        if self.diag:
            _, log_scores, _, indices, _ = self.cal(probabilities, indices)
        else:
            _, log_scores, _, indices = self.cal(probabilities, indices)

        nll = F.nll_loss(F.log_softmax(log_scores, dim=1), labels)
        loss = nll
        if regularize and self.diag:
            reg_loss, _ = self._compute_codeword_regularization_loss(
                indices, self.cal.T_code
            )
            loss = loss + reg_loss
        return loss, nll

    def fit_full_batch(
        self,
        train_loader,
        val_loader,
        device="cuda",
        max_steps=500,
        patience=10,
        min_delta=0.0,
    ):
        """Fit with one complete-dataset LBFGS update per outer iteration."""
        self.to(device)
        train_data = self._collect_full_dataset(train_loader, device)
        val_data = self._collect_full_dataset(val_loader, device)
        optimizer = self.configure_optimizers()

        best_val_nll = float("inf")
        best_state = None
        stale_steps = 0

        for step in range(max_steps):
            def closure():
                optimizer.zero_grad()
                loss, _ = self._full_batch_loss(*train_data, regularize=True)
                loss.backward()
                return loss

            optimizer.step(closure)

            with torch.no_grad():
                train_loss, train_nll = self._full_batch_loss(
                    *train_data, regularize=True
                )
                _, val_nll = self._full_batch_loss(*val_data, regularize=False)

            train_loss_value = train_loss.item()
            train_nll_value = train_nll.item()
            val_nll_value = val_nll.item()
            print(
                f"LBFGS step {step + 1}/{max_steps}: "
                f"train loss={train_loss_value:.6f}, "
                f"train NLL={train_nll_value:.6f}, "
                f"val NLL={val_nll_value:.6f}"
            )

            if val_nll_value < best_val_nll - min_delta:
                best_val_nll = val_nll_value
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in self.state_dict().items()
                }
                stale_steps = 0
            else:
                stale_steps += 1
                if stale_steps >= patience:
                    print(
                        "Early stopping LBFGS after "
                        f"{step + 1} steps; best validation NLL: "
                        f"{best_val_nll:.6f}"
                    )
                    break

        if best_state is None:
            raise RuntimeError("LBFGS fitting did not produce a valid model state.")

        self.load_state_dict(best_state, strict=True)
        return best_state, best_val_nll

    def configure_optimizers(self):
        """Build LBFGS; outer steps permit validation-based early stopping."""
        optimizer_kwargs = {
            "lr": self.optimizer_cfg.lr,
            "max_iter": 1,
            "max_eval": self.optimizer_cfg.get("max_eval", 1),
            "tolerance_grad": self.optimizer_cfg.get("tolerance_grad", 1e-7),
            "tolerance_change": self.optimizer_cfg.get("tolerance_change", 1e-9),
            "history_size": self.optimizer_cfg.get("history_size", 100),
            "line_search_fn": self.optimizer_cfg.get("line_search_fn", None), #"strong_wolfe"
        }
        return optim.LBFGS(self.cal.parameters(), **optimizer_kwargs)
                
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
        if self.finetune_only:
            logits = self.vqclassifier.cls(z)             # (B,C)
        else:
            logits = self.vqclassifier.cls(z_q)             # (B,C)        
        p_hat = F.softmax(logits, dim=1)
        
        if self.diag:
            calibrated_probs, log_scores, alpha, indices, T = self.cal(p_hat, indices)
        else:
            calibrated_probs, log_scores, alpha, indices = self.cal(p_hat, indices)
                
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
        elif self.finetune_only:     
            preds = logits.argmax(dim=1).view(-1,1)             
            out = {
                    "features": z.view(B, -1), #z_q.view(B, -1),               # (B, S*d), # quantized features alternatively use original features  
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
        
        
