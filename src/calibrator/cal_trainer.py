import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from utils.utils import *
from calibrator.local_net import *


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

def categorical_cross_entropy(probs, targets, eps=1e-8):
    probs = torch.clamp(probs, eps, 1.0 - eps)  # avoid log(0)
    return -torch.sum(targets * torch.log(probs), dim=1).mean()

def compute_multiclass_kl_divergence(p2, p1, num_classes, eps=1e-4):
    # Clip values to avoid log(0)
    p2 = torch.clamp(p2, eps, 1 - eps)
    p1 = torch.clamp(p1, eps, 1 - eps)
    m = 0.5 * (p1 + p2)
    m = torch.clamp(m, eps, 1 - eps)

    # KL divergence terms
    kl_p1_m = p1 * torch.log(p1 / m) + (1 - p1) * torch.log((1 - p1) / (1 - m))
    kl_p2_m = p2 * torch.log(p2 / m) + (1 - p2) * torch.log((1 - p2) / (1 - m))

    # Jensen-Shannon divergence
    js_divergence = 0.5 * (kl_p1_m + kl_p2_m)
    js_divergence = torch.maximum(js_divergence, torch.tensor(eps, device=js_divergence.device))

    # JS distance
    js_dist = torch.sqrt(js_divergence)

    # Entropy-based weighting (currently unused, but placeholder kept)
    weights = 1.0

    if num_classes == 2:
        to_ret = torch.sum(js_dist * weights, dim=0, keepdim=True)[0]
    else:
        to_ret = torch.mean(torch.mean(js_dist * weights, dim=0, keepdim=True), dim=1)[0]

    return to_ret
"""
def compute_multiclass_js_dist(p2, p1, num_classes, eps=1e-4):
    # Clip values to avoid log(0)
    p2 = torch.clamp(p2, eps, 1 - eps)
    p1 = torch.clamp(p1, eps, 1 - eps)
    m = 0.5 * (p1 + p2)
    m = torch.clamp(m, eps, 1 - eps)

    # KL divergence terms
    kl_p1_m = p1 * torch.log(p1 / m) + (1 - p1) * torch.log((1 - p1) / (1 - m))
    kl_p2_m = p2 * torch.log(p2 / m) + (1 - p2) * torch.log((1 - p2) / (1 - m))

    # Jensen-Shannon divergence
    js_divergence = 0.5 * (kl_p1_m + kl_p2_m)
    js_divergence = torch.maximum(js_divergence, torch.tensor(eps, device=js_divergence.device))

    # JS distance
    js_dist = torch.sqrt(js_divergence)

    if num_classes == 2:
        to_ret = torch.sum(js_dist, dim=0, keepdim=True)[0]
    else:
        to_ret = torch.mean(torch.mean(js_dist, dim=0, keepdim=True), dim=1)[0]

    return to_ret
"""    
# def multiclass_neighborhood_class0_prob(means, z_hat, sigma, y, eps=1e-6, ret_weights=False, test=False):
#     """
#     means: (B, C)
#     z_hat: (B, C)
#     sigma: (B, C) — variance per dimension (not stddev)
#     y: (B, C) — one-hot encoded label vector

#     Returns:
#         probs_classes: (B, C) — estimated P(y=c | Neigh_i) for each sample i
#     """
#     B = z_hat.shape[0]

#     # Expand dims for broadcasting
#     means_i = means.unsqueeze(1)  # (B, 1, C)
#     z_j = z_hat.unsqueeze(0)      # (1, B, C)

#     # Compute squared differences
#     diffs_squared = (z_j - means_i) ** 2  # (B, B, C)
    
#     # # Cancel self-similarity
#     # sq_dists = torch.sum(diffs_squared, dim=-1, keepdim=False)  # (B, B)
#     # eps_self = 1e-12
#     # mask = (sq_dists < eps_self).float()   # (B, N)    

#     # Expand sigma for broadcasting
#     sigma_i = sigma.unsqueeze(1)  # (B, 1, C)

#     # Gaussian kernel exponent
#     exponent = -diffs_squared / (2.0 * sigma_i + eps)  # (B, B, C)

#     # Sum over dimensions
#     K = torch.sum(exponent, dim=-1, keepdim=False)  # (B, B)
    
#     # Unnormalized similarity scores
#     p_unnorm = torch.exp(K)  # (B, B)    

#     if not test:
#         # Mask out self-contributions
#         mask = 1.0 - torch.eye(B, device=p_unnorm.device)
#         p_unnorm = p_unnorm * (1.0 - mask)

#     # Normalize across neighbors
#     p_norm = p_unnorm / (torch.sum(p_unnorm, dim=1, keepdim=True) + eps)
#     p_norm = torch.clamp(p_norm, eps, 1.0 - eps)

#     # Weighted sum over neighbors
#     probs_classes = torch.matmul(p_norm, y)  # (B, C)
#     if ret_weights:
#         return probs_classes, p_norm
#     else:
#         return probs_classes


def multiclass_neighborhood_class0_prob(means, z_hat, sigma, y, eps=1e-6, ret_weights=False, test=False):
    """
    means: (B, C)
    z_hat: (B, C)
    sigma: (B, C) — variance per dimension (not stddev)
    y: (B, C) — one-hot encoded label vector

    Returns:
        probs_classes: (B, C) — estimated P(y=c | Neigh_i) for each sample i
    """
    B = z_hat.shape[0]

    # Expand dims for broadcasting
    means_i = means.unsqueeze(1)  # (B, 1, C)
    z_j = z_hat.unsqueeze(0)      # (1, B, C)

    # Compute squared differences
    diffs_squared = (z_j - means_i) ** 2  # (B, B, C)

    # Expand sigma for broadcasting
    sigma_i = sigma.unsqueeze(1)  # (B, 1, C)

    # Gaussian kernel exponent
    exponent = -diffs_squared / (2.0 * sigma_i + eps)  # (B, B, C)

    # Sum over dimensions
    sq_dists = torch.sum(exponent, dim=-1, keepdim=False)  # (B, B)

    # Unnormalized similarity scores
    p_unnorm = torch.exp(sq_dists)  # (B, B)

    if not test:
        # Mask out self-contributions
        mask = 1.0 - torch.eye(B, device=p_unnorm.device)
        p_unnorm = p_unnorm * mask

    # Normalize across neighbors
    p_norm = p_unnorm / (torch.sum(p_unnorm, dim=1, keepdim=True) + eps)
    p_norm = torch.clamp(p_norm, eps, 1.0 - eps)

    # Weighted sum over neighbors
    probs_classes = torch.matmul(p_norm, y)  # (B, C)
    if ret_weights:
        return probs_classes, p_norm
    else:
        return probs_classes


def entropy(probs: torch.Tensor) -> torch.Tensor:
    """
    Computes entropy for a batch of probability distributions.
    
    Args:
        probs: Tensor of shape (N, n), rows are probability distributions.
    Returns:
        ent: Tensor of shape (N,), entropy of each row.
    """
    eps = 1e-12  # to avoid log(0)
    ent = -torch.sum(probs * torch.log(probs + eps), dim=1)
    return ent

class AuxTrainer(pl.LightningModule):
    def __init__(self, kwargs, num_classes):              
        super().__init__()
        self.loss_name = kwargs.loss.name
        if self.loss_name == 'focal':
            self.loss_fn = FocalLoss(gamma=kwargs.loss.gamma)
        self.model = AuxiliaryMLP(hidden_dim=kwargs.hidden_dim, latent_dim=num_classes, log_var_initializer=kwargs.log_var_initializer, dropout_rate=kwargs.dropout)
        self.num_classes = num_classes        
        self.optimizer_cfg = kwargs.optimizer
        self.init_alpha1 = kwargs.alpha1
        self.alpha1 = kwargs.alpha1
        self.alpha2 = kwargs.alpha2
        self.init_lambda_kl = kwargs.lambda_kl
        self.lambda_kl = kwargs.lambda_kl
        self.log_var_initializer = kwargs.log_var_initializer
        self.entropy_factor = kwargs.entropy_factor
        self.noise = kwargs.noise
        self.smoothing = kwargs.smoothing
        self.logits_scaling = kwargs.logits_scaling
        self.sampling = kwargs.sampling
        self.predict_labels = kwargs.predict_labels
        self.use_empirical_freqs = kwargs.use_empirical_freqs
        self.js_distance = kwargs.js_distance   
        self.interpolation_epochs = kwargs.interpolation_epochs     

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):                
        init_logits, y_one_hot, init_preds, init_preds_one_hot = batch        

        # Add noise
        epsilon = torch.randn_like(init_logits)
        noisy_logits = init_logits + self.noise * epsilon

        # Optional label smoothing        
        noisy_y_one_hot = label_smoothing(y_one_hot, self.smoothing) if self.smoothing else y_one_hot #random_label_smoothing

        # Forward pass
        latents = self(noisy_logits)
        means = latents[:, :self.num_classes]
        log_std = latents[:, self.num_classes:]
        stddev = F.softplus(log_std)
        sigma = stddev ** 2
        avg_variance = torch.mean(sigma) #dim=0

        # Reparameterization
        epsilon = torch.randn_like(means)
        z_hat = means + stddev * epsilon if self.sampling else means

        # Scaled probabilities
        probs_hat = F.softmax(z_hat / self.logits_scaling, dim=1)
        p2 = probs_hat

        # Neighborhood-based probabilities
        p1 = multiclass_neighborhood_class0_prob(means, z_hat, sigma=sigma, y=noisy_y_one_hot) 

        # KL divergence
        kl_loss = compute_multiclass_kl_divergence(p2, p1, self.num_classes)

        # Constraint loss
        new_preds = torch.argmax(probs_hat, dim=1)
        prediction_mask = (new_preds == init_preds).float()
        constraint = torch.mean(1.0 - prediction_mask)

        if self.predict_labels:
            target = y_one_hot #noisy_y_one_hot 
        else:
            target = init_preds_one_hot 
        if self.use_empirical_freqs:
            scores = p1
        else:
            scores = p2
            
        if self.loss_name == 'focal':
            constraint_loss = self.loss_fn(scores, target) # focal variant of opur loss
        else:    
            constraint_loss = categorical_cross_entropy(scores, target) #F.cross_entropy(quantity, torch.argmax(target, dim=1))
            
        if self.current_epoch < self.interpolation_epochs:
            self.interpolate_weights()

        total_loss = (self.lambda_kl * kl_loss +
                      self.alpha1 * constraint_loss +
                      self.alpha2 * avg_variance ** 2)

        self.log("train_total", total_loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("train_kl", kl_loss, on_epoch=True, on_step=False, prog_bar=False)
        self.log("train_con_loss", constraint_loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("train_constraint", constraint, on_epoch=True, on_step=False, prog_bar=False)
        #self.log("lambda_kl", self.lambda_kl, on_epoch=True, on_step=False, prog_bar=False)

        return total_loss
    
    def validation_step(self, batch):
        init_logits, y_one_hot, init_preds, init_preds_one_hot = batch

        # Add noise
        epsilon = torch.randn_like(init_logits)
        noisy_logits = init_logits + self.noise * epsilon

        # Optional label smoothing        
        noisy_y_one_hot = label_smoothing(y_one_hot, self.smoothing) if self.smoothing else y_one_hot

        # Forward pass
        latents = self(noisy_logits)
        means = latents[:, :self.num_classes]
        log_std = latents[:, self.num_classes:]
        stddev = F.softplus(log_std)
        sigma = stddev ** 2
        avg_variance = torch.mean(sigma)

        # Reparameterization
        epsilon = torch.randn_like(means)
        z_hat = means + stddev * epsilon if self.sampling else means

        # Scaled probabilities
        probs_hat = F.softmax(z_hat / self.logits_scaling, dim=1)
        p2 = probs_hat

        # Neighborhood-based probabilities
        p1 = multiclass_neighborhood_class0_prob(means, z_hat, sigma=sigma, y=noisy_y_one_hot) 

        # KL divergence
        kl_loss = compute_multiclass_kl_divergence(p2, p1, self.num_classes)

        # Constraint loss
        new_preds = torch.argmax(probs_hat, dim=1)
        prediction_mask = (new_preds == init_preds).float()
        constraint = torch.mean(1.0 - prediction_mask)

        if self.predict_labels:
            target = y_one_hot 
        else:
            target = init_preds_one_hot 
        if self.use_empirical_freqs:
            scores = p1
        else:
            scores = p2
        
        if self.loss_name == 'focal':
            constraint_loss = self.loss_fn(scores, target) #F.cross_entropy(quantity, torch.argmax(target, dim=1))
        else:    
            constraint_loss = categorical_cross_entropy(scores, target) #F.cross_entropy(quantity, torch.argmax(target, dim=1))
        
        total_loss = (self.lambda_kl * kl_loss +
                      self.alpha1 * constraint_loss +
                      self.alpha2 * avg_variance ** 2)              
        optuna_loss = kl_loss + constraint_loss 
            
        self.log("val_total", total_loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("optuna_loss", optuna_loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("val_kl", kl_loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("val_con_loss", constraint_loss, on_epoch=True, on_step=False, prog_bar=False)
        self.log("val_constraint", constraint, on_epoch=True, on_step=False, prog_bar=True)  
        self.log("lambda_kl", self.lambda_kl, on_epoch=True, on_step=False, prog_bar=False)       
        self.log("alpha1", self.alpha1, on_epoch=True, on_step=False, prog_bar=False)       
        self.log("log_var", self.log_var_initializer, on_epoch=True, on_step=False, prog_bar=False)                              

    def configure_optimizers(self):
        opt_name = self.optimizer_cfg.name #.lower()
        opt_name = opt_name[0].upper() + opt_name[1:]
        lr = self.optimizer_cfg.lr
        wd = self.optimizer_cfg.get("weight_decay", 0.0)        
        
        # Dynamically get the optimizer class
        optimizer_class = getattr(torch.optim, opt_name, None)
        if optimizer_class is None:
            raise ValueError(f"Unsupported optimizer: {opt_name}")

        # Build kwargs dynamically
        optimizer_kwargs = {"lr": lr, "weight_decay": wd}
        if opt_name == "SGD":
            optimizer_kwargs["momentum"] = self.optimizer_cfg.get("momentum", 0.9)

        optimizer = optimizer_class(self.parameters(), **optimizer_kwargs)
        return optimizer
    
    def predict_step(self, batch):
        """
        Prediction step for a batch of data.
        :param batch:
        A tuple containing the input data `x`, target labels `y`, and feedback `h`.
        :param batch_idx:
        Index of the batch in the current epoch.
        :param dataloader_idx:
        Index of the dataloader (default is 0).
        :return:
            A dictionary containing the predictions, probabilities, feedback, true labels, rejection score, and selection status.
        """
        init_logits, y_one_hot, _, _ = batch        
        
        # Add noise
        epsilon = torch.randn_like(init_logits)
        noisy_logits = init_logits + self.noise * epsilon
        
        # Forward pass
        latents = self(noisy_logits)
        means = latents[:, :self.num_classes]
        log_std = latents[:, self.num_classes:]
        stddev = F.softplus(log_std)
        
        # Reparameterization
        epsilon = torch.randn_like(means)
        new_logits = means + stddev * epsilon if self.sampling else means
        
        # Scaled probabilities        
        preds = torch.argmax(new_logits, dim=-1).view(-1,1)
        target = torch.argmax(y_one_hot, dim=-1).view(-1,1)
        return {
            "preds": preds,            
            "true": target,
            "logits": new_logits,
        }
    
    def interpolate_weights(self):
        epoch = self.current_epoch
        total_epochs = self.interpolation_epochs
        start_ratio = self.init_lambda_kl
        end_ratio = 10 #self.init_alpha1
        ratio = start_ratio + (end_ratio - start_ratio) * (epoch / total_epochs)
        self.lambda_kl = ratio
        self.alpha1 = end_ratio
       


class AuxTrainerV2(pl.LightningModule):
    def __init__(self, kwargs, num_classes, feature_dim, similarity_dim):              
        super().__init__()        
        self.loss_name = kwargs.loss.name
        if self.loss_name == 'focal':
            self.loss_fn = FocalLoss(gamma=kwargs.loss.gamma)                            
        self.num_classes = num_classes        
        self.optimizer_cfg = kwargs.optimizer
        self.init_alpha1 = kwargs.alpha1
        self.alpha1 = kwargs.alpha1
        self.alpha2 = kwargs.alpha2
        self.init_lambda_kl = kwargs.lambda_kl
        self.lambda_kl = kwargs.lambda_kl
        self.log_var_initializer = kwargs.log_var_initializer
        self.entropy_factor = kwargs.entropy_factor
        self.noise = kwargs.noise
        self.smoothing = kwargs.smoothing
        self.logits_scaling = kwargs.logits_scaling
        self.sampling = kwargs.sampling
        self.predict_labels = kwargs.predict_labels
        self.use_empirical_freqs = kwargs.use_empirical_freqs
        self.js_distance = kwargs.js_distance   
        self.interpolation_epochs = kwargs.interpolation_epochs     
        self.feature_dim = feature_dim
        self.similarity_dim = similarity_dim
        self.adabw = kwargs.adabw         
        self.fixed_var = kwargs.fixed_var
        self.linearly_combine_pca = kwargs.linearly_combine_pca
        self.kernel_only = kwargs.kernel_only
        self.device_name = 'cuda'   
        self.augment = kwargs.augment 
        self.init_alpha_sim = kwargs.alpha_sim
        self.init_alpha_cls = kwargs.alpha_cls    

        self.model = AuxiliaryMLPV2(hidden_dim=kwargs.hidden_dim, feature_dim=feature_dim, output_dim=num_classes, 
                                    similarity_dim=similarity_dim, init_alpha_sim=self.init_alpha_sim, init_alpha_cls=self.init_alpha_cls, 
                                    log_var_initializer=kwargs.log_var_initializer, dropout_rate=kwargs.dropout, 
                                    fixed_var=self.fixed_var, linearly_combine_pca=kwargs.linearly_combine_pca)
        
        
    def forward(self, init_feats, init_logits, init_pca):
        return self.model(init_feats, init_logits, init_pca)

    
    def training_step(self, batch):                
        init_feats, init_logits, init_pca, y_one_hot, init_preds, init_preds_one_hot = batch        

        # standardize per dimension (across batch axis, i.e. dim=0)
        #mean = init_pca.mean(dim=0, keepdim=True)
        #std = init_pca.std(dim=0, keepdim=True)

        # avoid divide by zero
        #init_pca = (init_pca - mean) / (std + 1e-8)
        
        # Add noise
        epsilon = torch.randn_like(init_feats)
        if self.augment:
            noisy_feats = init_feats + self.noise * epsilon
        else:
            noisy_feats = init_feats

        # Optional label smoothing        
        noisy_y_one_hot = label_smoothing(y_one_hot, self.smoothing) if self.smoothing else y_one_hot #random_label_smoothing

        # Forward pass
        latents_class, latents_sim = self(noisy_feats, init_logits, init_pca)
        means = latents_sim[:, :self.similarity_dim] #init_pcas        
        if self.fixed_var:
            var_tensor = torch.full((1,), self.log_var_initializer)
            var_tensor = torch.log(torch.exp(var_tensor) - 1)       
            stddev = F.softplus(var_tensor).to(self.device_name)
        else:
            log_std = latents_sim[:, self.similarity_dim:] #latents_sim
            stddev = F.softplus(log_std)
            
        sigma = stddev ** 2
        avg_variance = torch.mean(sigma) #dim=0
        min_variance = torch.min(sigma)
        max_variance = torch.max(sigma)
        pca_std = means.std(dim=0, keepdim=True).mean()

        # Reparameterization
        epsilon = torch.randn_like(means)
        z_hat = means + stddev * epsilon if self.sampling else means

        # Scaled probabilities
        probs_hat = F.softmax(latents_class / self.logits_scaling, dim=1)
        p2 = probs_hat

        # Neighborhood-based probabilities
        p1, weights = multiclass_neighborhood_class0_prob(means, z_hat, sigma=sigma, y=noisy_y_one_hot, ret_weights=True) # compute on similarity specific encoding!!!        
                  
        # KL divergence
        kl_loss = compute_multiclass_kl_divergence(p2, p1, self.num_classes)

        # Constraint loss
        new_preds = torch.argmax(probs_hat, dim=1)
        prediction_mask = (new_preds == init_preds).float()
        constraint = torch.mean(1.0 - prediction_mask)

        if self.predict_labels:
            target = y_one_hot #noisy_y_one_hot 
        else:
            target = init_preds_one_hot 
        if self.use_empirical_freqs:
            scores = p1
        else:
            scores = p2
            
        if self.loss_name == 'focal':
            constraint_loss = self.loss_fn(scores, target) # focal variant of opur loss
        else:    
            constraint_loss = categorical_cross_entropy(scores, target) #F.cross_entropy(quantity, torch.argmax(target, dim=1))
            
        if self.current_epoch < self.interpolation_epochs:
            self.interpolate_weights()

        total_loss = (self.lambda_kl * kl_loss +
                      self.alpha1 * constraint_loss +
                      self.alpha2 * entropy(weights).mean()) #avg_variance pca_std)

        self.log("train_total", total_loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("train_kl", kl_loss, on_epoch=True, on_step=False, prog_bar=False)
        self.log("train_con_loss", constraint_loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("train_constraint", constraint, on_epoch=True, on_step=False, prog_bar=False)
        self.log('avg_variance', avg_variance, on_epoch=True, on_step=False, prog_bar=False)
        self.log('min_variance', min_variance, on_epoch=True, on_step=False, prog_bar=False)
        self.log('max_variance', max_variance, on_epoch=True, on_step=False, prog_bar=False)
        self.log('pca_std', pca_std, on_epoch=True, on_step=False, prog_bar=False)
        #self.log("lambda_kl", self.lambda_kl, on_epoch=True, on_step=False, prog_bar=False)

        return total_loss
    
    
    def validation_step(self, batch):
        init_feats, init_logits, init_pca, y_one_hot, init_preds, init_preds_one_hot = batch

        # standardize per dimension (across batch axis, i.e. dim=0)
        #mean = init_pca.mean(dim=0, keepdim=True)
        #std = init_pca.std(dim=0, keepdim=True)

        # avoid divide by zero
        #init_pca = (init_pca - mean) / (std + 1e-8)
        
        # Add noise
        epsilon = torch.randn_like(init_feats)
        if self.augment:
            noisy_feats = init_feats + self.noise * epsilon
        else:
            noisy_feats = init_feats

        # Optional label smoothing        
        noisy_y_one_hot = y_one_hot #label_smoothing(y_one_hot, self.smoothing) if self.smoothing else y_one_hot

        # Forward pass
        latents_class, latents_sim = self(noisy_feats, init_logits, init_pca)
        means = latents_sim[:, :self.similarity_dim]#init_pca        
        if self.fixed_var:
            var_tensor = torch.full((1,), self.log_var_initializer)
            var_tensor = torch.log(torch.exp(var_tensor) - 1)       
            stddev = F.softplus(var_tensor).to(self.device_name)       
        else:
            log_std = latents_sim[:, self.similarity_dim:] #latents_sim        
            stddev = F.softplus(log_std)
        sigma = stddev ** 2
        avg_variance = torch.mean(sigma)
        min_variance = torch.min(sigma)
        max_variance = torch.max(sigma)
        pca_std = means.std(dim=0, keepdim=True).mean()

        # Reparameterization
        epsilon = torch.randn_like(means)
        z_hat = means + stddev * epsilon if self.sampling else means

        # Scaled probabilities
        probs_hat = F.softmax(latents_class / self.logits_scaling, dim=1)
        p2 = probs_hat

        # Neighborhood-based probabilities
        p1, weights = multiclass_neighborhood_class0_prob(means, z_hat, sigma=sigma, y=noisy_y_one_hot, ret_weights=True)             

        # KL divergence
        kl_loss = compute_multiclass_kl_divergence(p2, p1, self.num_classes)

        # Constraint loss
        new_preds = torch.argmax(probs_hat, dim=1)
        prediction_mask = (new_preds == init_preds).float()
        constraint = torch.mean(1.0 - prediction_mask)

        if self.predict_labels:
            target = y_one_hot 
        else:
            target = init_preds_one_hot 
        if self.use_empirical_freqs:
            scores = p1
        else:
            scores = p2
        
        if self.loss_name == 'focal':
            constraint_loss = self.loss_fn(scores, target) #F.cross_entropy(quantity, torch.argmax(target, dim=1))
        else:    
            constraint_loss = categorical_cross_entropy(scores, target) #F.cross_entropy(quantity, torch.argmax(target, dim=1))
        
        total_loss = (self.lambda_kl * kl_loss +
                      self.alpha1 * constraint_loss +
                      self.alpha2 * entropy(weights).mean()) #avg_variance pca_std)              
        optuna_loss = kl_loss + constraint_loss 
            
        self.log("val_total", total_loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("optuna_loss", optuna_loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("val_kl", kl_loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("val_con_loss", constraint_loss, on_epoch=True, on_step=False, prog_bar=False)
        self.log("val_constraint", constraint, on_epoch=True, on_step=False, prog_bar=True)  
        self.log('avg_val_variance', avg_variance, on_epoch=True, on_step=False, prog_bar=False)    
        self.log('min_val_variance', min_variance, on_epoch=True, on_step=False, prog_bar=False)
        self.log('max_val_variance', max_variance, on_epoch=True, on_step=False, prog_bar=False)
        self.log('val_pca_std', pca_std, on_epoch=True, on_step=False, prog_bar=False)
        #self.log("lambda_kl", self.lambda_kl, on_epoch=True, on_step=False, prog_bar=False)       
        #self.log("alpha1", self.alpha1, on_epoch=True, on_step=False, prog_bar=False)       
        #self.log("log_var", self.log_var_initializer, on_epoch=True, on_step=False, prog_bar=False)    
                              
    """
    def training_step(self, batch):                
        init_feats, init_logits, init_pca, y_one_hot, init_preds, init_preds_one_hot = batch        

        # Optional label smoothing        
        noisy_y_one_hot = label_smoothing(y_one_hot, self.smoothing) if self.smoothing else y_one_hot 

        # Forward pass
        latents_class, latents_sim = self(init_feats, init_logits, init_pca)
        means = latents_sim[:, :self.similarity_dim]               
        var_tensor = torch.full((1,), self.log_var_initializer)
        var_tensor = torch.log(torch.exp(var_tensor) - 1)       
        stddev = F.softplus(var_tensor).to(self.device_name)        
            
        sigma = stddev ** 2        
        z_hat = means

        # Probabilities
        probs_hat = F.softmax(latents_class, dim=1)
        p2 = probs_hat

        # Neighborhood-based probabilities
        p1, weights = multiclass_neighborhood_class0_prob(means, z_hat, sigma=sigma, y=noisy_y_one_hot, ret_weights=True) 

        # JS distance
        js_loss = compute_multiclass_js_dist(p2, p1, self.num_classes)
        
        target = y_one_hot        
        scores = p1
            
        constraint_loss = categorical_cross_entropy(scores, target)
            
        total_loss = (self.lambda_kl * js_loss +
                      self.alpha1 * constraint_loss)

        self.log("train_total", total_loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("train_js", js_loss, on_epoch=True, on_step=False, prog_bar=False)
        self.log("train_con_loss", constraint_loss, on_epoch=True, on_step=False, prog_bar=True)

        return total_loss
    
    def validation_step(self, batch):
        init_feats, init_logits, init_pca, y_one_hot, init_preds, init_preds_one_hot = batch                

        # Optional label smoothing        
        noisy_y_one_hot = label_smoothing(y_one_hot, self.smoothing) if self.smoothing else y_one_hot

        # Forward pass
        latents_class, latents_sim = self(init_feats, init_logits, init_pca)
        means = latents_sim[:, :self.similarity_dim]        
        var_tensor = torch.full((1,), self.log_var_initializer)
        var_tensor = torch.log(torch.exp(var_tensor) - 1)       
        stddev = F.softplus(var_tensor).to(self.device_name)       
        sigma = stddev ** 2

        z_hat = means

        # Probabilities
        probs_hat = F.softmax(latents_class, dim=1)
        p2 = probs_hat

        # Neighborhood-based probabilities
        p1, weights = multiclass_neighborhood_class0_prob(means, z_hat, sigma=sigma, y=noisy_y_one_hot, ret_weights=True) 

        # JS distance
        js_loss = compute_multiclass_js_dist(p2, p1, self.num_classes)

        target = y_one_hot 
        scores = p1    
            
        constraint_loss = categorical_cross_entropy(scores, target) 
        
        total_loss = (self.lambda_kl * js_loss +
                      self.alpha1 * constraint_loss) 
        optuna_loss = js_loss + constraint_loss 
            
        self.log("val_total", total_loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("optuna_loss", optuna_loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("val_js", js_loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("val_con_loss", constraint_loss, on_epoch=True, on_step=False, prog_bar=False)
        
"""        
    def configure_optimizers(self):
        opt_name = self.optimizer_cfg.name #.lower()
        opt_name = opt_name[0].upper() + opt_name[1:]
        lr = self.optimizer_cfg.lr
        wd = self.optimizer_cfg.get("weight_decay", 0.0)        
        
        # Dynamically get the optimizer class
        if 'dam' in opt_name:
            optimizer_class = getattr(torch.optim, opt_name, None)
        elif 'gd' in opt_name:
            optimizer_class = getattr(torch.optim, 'SGD', None)
        
        if optimizer_class is None:
            raise ValueError(f"Unsupported optimizer: {opt_name}")

        # Build kwargs dynamically
        optimizer_kwargs = {"lr": lr, "weight_decay": wd}
        if opt_name == "SGD":
            optimizer_kwargs["momentum"] = self.optimizer_cfg.get("momentum", 0.9)

        optimizer = optimizer_class(self.parameters(), **optimizer_kwargs)
        return optimizer
    
    
    def predict_step(self, batch):
        """
        Prediction step for a batch of data.
        :param batch:
        A tuple containing the input data `x`, target labels `y`, and feedback `h`.
        :param batch_idx:
        Index of the batch in the current epoch.
        :param dataloader_idx:
        Index of the dataloader (default is 0).
        :return:
            A dictionary containing the predictions, probabilities, feedback, true labels, rejection score, and selection status.
        """
        init_feats, init_logits, init_pca, y_one_hot, init_preds, init_preds_one_hot = batch 
        
        # standardize per dimension (across batch axis, i.e. dim=0)
        #mean = init_pca.mean(dim=0, keepdim=True)
        #std = init_pca.std(dim=0, keepdim=True)

        # avoid divide by zero
        #init_pca = (init_pca - mean) / (std + 1e-8)
        
        # Add noise
        epsilon = torch.randn_like(init_feats)
        if self.augment:
            noisy_feats = init_feats + self.noise * epsilon
        else:
            noisy_feats = init_feats
        
        # Forward pass
        latents_class, latents_sim = self(noisy_feats, init_logits, init_pca)
        means = latents_sim[:, :self.similarity_dim] #init_pca
        
        if self.fixed_var:
            var_tensor = torch.full((1,), self.log_var_initializer)
            var_tensor = torch.log(torch.exp(var_tensor) - 1)       
            stddev = F.softplus(var_tensor).to(self.device_name)     
        else:            
            log_std = latents_sim[:, self.similarity_dim:] #latents_sim
            stddev = F.softplus(log_std)
        
        # Reparameterization
        epsilon = torch.randn_like(means)
        new_logits = latents_class + stddev * epsilon if self.sampling else latents_class
        
        # Scaled probabilities        
        preds = torch.argmax(latents_class, dim=-1).view(-1,1)
        target = torch.argmax(y_one_hot, dim=-1).view(-1,1)
                
        return {
            "preds": preds,            
            "true": target,
            "logits": new_logits,
        }                    


    def build_calibration_memory(self, calibration_data):
        """
        Iterates through self.calibration_data and stores:
        - z_cal:   all similarity latents (N_cal, D)
        - y_cal:   all one-hot labels     (N_cal, C)
        """

        latents_list = []
        labels_list  = []

        with torch.no_grad():
            for batch in calibration_data:
                init_feats, init_logits, init_pca, y_one_hot, init_preds, init_preds_one_hot = batch

                # init_feats = init_feats.to(self.device_name)
                # init_logits = init_logits.to(self.device_name)
                # init_pca = init_pca.to(self.device_name)
                # y_one_hot = y_one_hot.to(self.device_name)

                # Add noise if needed
                epsilon = torch.randn_like(init_feats)
                if self.augment:
                    noisy_feats = init_feats + self.noise * epsilon
                else:
                    noisy_feats = init_feats

                # Forward pass
                _, latents_sim = self(noisy_feats, init_logits, init_pca)

                # Similarity features (z_cal)
                z_cal = latents_sim[:, :self.similarity_dim]

                latents_list.append(z_cal)
                labels_list.append(y_one_hot)

        # Concatenate and ensure they live on the same device as everything else
        self.z_cal_full = torch.cat(latents_list, dim=0).to(self.device_name)  # (N_cal, D)
        self.y_cal_full = torch.cat(labels_list, dim=0).to(self.device_name)   # (N_cal, C)
        
        # === NEW: estimate bandwidth from z_cal_full ===
        h = estimate_bandwidth_silverman(self.z_cal_full)  # (D,)
        
        # store either per-dim or a global scalar
        self.bandwidth_vec = h.to(self.device_name)        # std per dim
        self.bandwidth_scalar = float(h.mean().item())     # optional global std        
        print('OPTIMAL GAMMA FOUND: ', self.bandwidth_scalar)

        
    def kernel_predict_step(self, batch):
        """
        Prediction step for a batch of data.
        :param batch:
        A tuple containing the input data `x`, target labels `y`, and feedback `h`.
        :param batch_idx:
        Index of the batch in the current epoch.
        :param dataloader_idx:
        Index of the dataloader (default is 0).
        :return:
            A dictionary containing the predictions, probabilities, feedback, true labels, rejection score, and selection status.
        """
        init_feats, init_logits, init_pca, y_one_hot, _, _ = batch
        
        init_feats = init_feats.to(self.device_name)
        init_logits = init_logits.to(self.device_name)
        init_pca = init_pca.to(self.device_name)
        y_one_hot = y_one_hot.to(self.device_name)

        # -------- Forward for test batch --------
        epsilon = torch.randn_like(init_feats)
        if self.augment:
            noisy_feats = init_feats + self.noise * epsilon
        else:
            noisy_feats = init_feats

        _, latents_sim = self(noisy_feats, init_logits, init_pca)
        means = latents_sim[:, :self.similarity_dim]      # (B_test, D)

        if self.fixed_var:
            var_tensor = torch.full((1,), self.log_var_initializer, device=self.device_name)
            var_tensor = torch.log(torch.exp(var_tensor) - 1)
            stddev = F.softplus(var_tensor)
            sigma = stddev**2  #self.bandwidth_scalar**2 
        else:
            log_std = latents_sim[:, self.similarity_dim:]
            stddev = F.softplus(log_std)
            sigma = stddev**2 #self.bandwidth_scalar**2 
        
        # -------- Apply your original KDE function --------
        probs_classes = multiclass_neighborhood_class0_prob(
            means,                     # (B_test, D)
            self.z_cal_full,           # (N_cal, D)
            sigma,                     # (B_test, D) or scalar
            self.y_cal_full,           # (N_cal, C)
            test=True
        )            
        
        preds = torch.argmax(probs_classes, dim=-1).view(-1,1)                            
        target = torch.argmax(y_one_hot, dim=-1).view(-1,1)
                
        return {
            "preds": preds,            
            "true": target,
            "logits": probs_classes,
        }
        
    def extract_pca(self, batch):
        init_feats, init_logits, init_pca, y_one_hot, _, _ = batch 
        
        # standardize per dimension (across batch axis, i.e. dim=0)
        #mean = init_pca.mean(dim=0, keepdim=True)
        #std = init_pca.std(dim=0, keepdim=True)

        # avoid divide by zero
        #init_pca = (init_pca - mean) / (std + 1e-8)
        
        # Add noise
        epsilon = torch.randn_like(init_feats)
        if self.augment:
            noisy_feats = init_feats + self.noise * epsilon
        else:
            noisy_feats = init_feats
        
        # Forward pass
        logits, latents_sim = self(noisy_feats, init_logits, init_pca)
        means = latents_sim[:, :self.similarity_dim] #init_pca
        if self.fixed_var:
            var_tensor = torch.full((1,), self.log_var_initializer)
            var_tensor = torch.log(torch.exp(var_tensor) - 1)       
            stddev = F.softplus(var_tensor).to(self.device_name)
        else:            
            log_std = latents_sim[:, self.similarity_dim:] #latents_sim
            stddev = F.softplus(log_std)                
        sigma = stddev**2

        if self.lambda_kl == 0 or self.kernel_only: 
            # sigma = torch.full((1,), self.bandwidth_scalar**2, device=self.device_name) #stddev**2 #            
            # -------- Apply your original KDE function --------
            probs_classes = multiclass_neighborhood_class0_prob(
                means,                     # (B_test, D)
                self.z_cal_full,           # (N_cal, D)
                sigma,                     # (B_test, D) or scalar
                self.y_cal_full,           # (N_cal, C)
                test=True
            )    
                  
            # Ensure valid probability range
            probs_classes = torch.clamp(probs_classes, 0.0, 1.0)  
            
            preds = torch.argmax(probs_classes, dim=-1).view(-1,1)                            
        
            # Create dict in the same format as predict outputs
            if self.adabw:
                out = {
                    "features": means, #init_pca      
                    "bandwidth": sigma,
                    "logits": probs_classes,
                    "preds": preds,     
                    "true": torch.argmax(y_one_hot, dim=-1).view(-1,1)
                }
            else:
                out = {
                    "features": means, #means, #init_pca       
                    "logits": probs_classes,
                    "preds": preds,     
                    "true": torch.argmax(y_one_hot, dim=-1).view(-1,1)
                }
        else:
            preds = torch.argmax(logits, dim=-1).view(-1,1)  # predicted class
            # Create dict in the same format as predict outputs
            if self.adabw:
                out = {
                    "features": means, #init_pca      
                    "bandwidth": sigma,
                    "logits": logits,
                    "preds": preds,     
                    "true": torch.argmax(y_one_hot, dim=-1).view(-1,1)
                }
            else:
                out = {
                    "features": means, #means, #init_pca       
                    "logits": logits,
                    "preds": preds,     
                    "true": torch.argmax(y_one_hot, dim=-1).view(-1,1)
                }
        return out
    
    
    def interpolate_weights(self):
        epoch = self.current_epoch
        total_epochs = self.interpolation_epochs
        start_ratio = self.init_lambda_kl
        end_ratio = 10 #self.init_alpha1
        ratio = start_ratio + (end_ratio - start_ratio) * (epoch / total_epochs)
        self.lambda_kl = ratio
        self.alpha1 = end_ratio
       

