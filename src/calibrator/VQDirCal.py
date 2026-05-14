import torch
import torch.nn as nn
import torch.nn.functional as F
import math 


def softplus_inv(y: float) -> float:
        # stable: log(exp(y) - 1)
        return math.log(math.expm1(y))
    
# log B(a) = sum_i lgamma(a_i) - lgamma(sum_i a_i)    
def log_multivariate_beta(alpha, dim=1):
    # alpha: (..., C) with alpha > 0
    return torch.lgamma(alpha).sum(dim=dim) - torch.lgamma(alpha.sum(dim=dim))

    
class VQDirCal(nn.Module):
    """
    Implements:
      W(x) = [a_{s(1)}^T; ...; a_{s(S)}^T] in R^{S x C}
      U(x) = [b_{s(1)}^T; ...; b_{s(S)}^T] in R^{S x C}
      A(x) = W(x)^T U(x) in R^{C x C}
      alpha^{(j)}(x) = softplus(A(x)[:, j]) + eps
      log P(y=j | p_hat, V) = b_j + (alpha^{(j)}-1)^T log p_hat + const
      b_j = log pi_j|V - log B(alpha^{(j)})
    As in Eqs. (30)-(34). :contentReference[oaicite:2]{index=2}
    """

    def __init__(self, K: int, C: int, S: int, eps: float = 1e-8, quadratic: bool = False, learn_pi: bool = False, learn_bias: bool=False, diag: bool = False,
                 random: bool = False, standard_dirichlet: bool = False): #1e-8
        super().__init__()
        self.K = K
        self.C = C
        self.S = S
        self.eps = eps
        self.diag = diag
        self.random = random
        self.standard_dirichlet = standard_dirichlet
        self.learn_bias = learn_bias
        self.quadratic = quadratic
        if standard_dirichlet:
            self.W = nn.Parameter(torch.zeros(C, C))
            self.b = nn.Parameter(torch.zeros(C))
            nn.init.normal_(self.W, mean=0., std=0.01)
            nn.init.normal_(self.b, mean=0., std=0.01) 
        else:    
            # Calibration codebooks A (receiver) and B (sender): K vectors in R^C
            # Use embeddings so indexing is fast: indices in [0..K-1]
            self.A_code = nn.Embedding(K, C)  # corresponds to {a_1,...,a_K} in R^C
            self.B_code = nn.Embedding(K, C)  # corresponds to {b_1,...,b_K} in R^C      
            if learn_bias:        
                self.bias_code = nn.Embedding(K, C)  # corresponds to {b_1,...,b_K} in R^C 
                nn.init.normal_(self.bias_code.weight, mean=0., std=0.01)       
            if self.diag or self.learn_bias:      
                self.T_code = nn.Parameter(torch.ones(S))  # corresponds to {b_1,...,b_K} in R^C     
                nn.init.normal_(self.T_code, mean=1., std=0.01)   
            if self.quadratic:
                self.Q = nn.Embedding(K, C)  # for quadratic terms
                self.P = nn.Embedding(K, C)  # for quadratic terms
                nn.init.normal_(self.Q.weight, mean=0., std=0.01)                
                nn.init.normal_(self.P.weight, mean=0., std=0.01)

            # Prior pi_{j|V}
            # PDF allows cell-dependent priors; a practical baseline is global pi.
            # If learn_pi=True, learn a global prior over classes.
            if learn_pi:
                self.pi_logits = nn.Parameter(torch.zeros(C))
            else:
                self.register_buffer("pi", torch.full((C,), 1.0 / C))

            # nn.init.normal_(self.A_code.weight, mean=0., std=0.01) 
            # nn.init.normal_(self.B_code.weight, mean=0., std=0.01) 
            self.init_alpha_minus_1_all_1s() # intitialises alpha_minus_1 as a matrix of all 1s   
            if hasattr(self, "pi_logits"):        
                # self.init_pi_to_cancel_logB()        
                nn.init.normal_(self.pi_logits, mean=0., std=0.01)                  

    @torch.no_grad()
    def set_global_pi_from_counts(self, y: torch.Tensor):
        """Estimate global pi from calibration labels y (B,)"""
        counts = torch.bincount(y.long(), minlength=self.C).float()
        self.pi = (counts / counts.sum().clamp_min(1.0)).to(self.pi.device)

    def forward(self, p_hat: torch.Tensor, indices: torch.Tensor):
        """
        p_hat:   (B, C) predicted probabilities from base classifier (softmax)
        indices: (B, S) VQ indices per slot (0..K-1)
        returns:
          post:   (B, C) calibrated posterior P(y | p_hat, V)
          scores: (B, C) unnormalised log posterior scores
          alpha:  (B, C, C) alpha_{i,j} (pred-class i, true-class j)
        """
        if self.standard_dirichlet:
            B, C = p_hat.shape
            assert C == self.C
            logp = torch.log(p_hat.clamp_min(self.eps))  # (B,C)
            linear_term = torch.matmul(logp, self.W)     # (B,C)
            b = self.b.unsqueeze(0).expand(B, -1)        # (B,C)
            log_scores = b + linear_term
            calibrated_probs = F.softmax(log_scores, dim=1)
            alpha = self.W.unsqueeze(0).repeat(B, 1, 1) 
            return calibrated_probs, log_scores, alpha
        
        else:
            B, C = p_hat.shape
            assert C == self.C
            assert indices.dim() == 2

            if self.random:            
                # Use random indices instead of VQ indices during training
                g = torch.Generator(device=indices.device)
                g.manual_seed(torch.seed())
                K = self.K
                S = self.S #indices.shape[1]
                indices = torch.randint(low=0, high=K, size=(B, S), generator=g, device=indices.device)            
                
            # Build W(x), U(x) by selecting vectors per slot: (B, S, C)
            # W rows are a_{s(i)}, U rows are b_{s(i)}  (Eq. 30) :contentReference[oaicite:3]{index=3}
            W = self.A_code(indices.long())  # (B,S,C)
            U = self.B_code(indices.long())  # (B,S,C)                  
            
            # A(x) = W^T U -> (B, C, C)   (Eq. 31) :contentReference[oaicite:4]{index=4}            
            if self.diag:
                T = self.T_code.view(1, self.S, 1)      # (1,S,1) for broadcasting
                U_w = U * T #(T**2)
            else:
                U_w = U
            A = torch.matmul(W.transpose(1, 2), U_w)                        
            
            # Positivity constraint for Dirichlet concentrations (PDF remark) :contentReference[oaicite:5]{index=5}            
            alpha = F.softplus(A) + 1e-8 #F.softplus(A).pow(self.tau) + 1e-8 #torch.exp(A / self.tau) + self.eps  # (B,C,C) + T.sum(dim=1).unsqueeze(-1)

            # alpha^{(j)} is column j: alpha[:, :, j] in (B,C)
            alpha_minus_1 = alpha - 1.0                                     # initialised as all 1s matrix
            I = torch.eye(C, device=alpha.device).unsqueeze(0)              # Identity matrix
            alpha_minus_1 = alpha_minus_1 + I # -1.0                        # make alpha_minus_1 initally identity matrix
            # import pdb; pdb.set_trace()
            
            # log p_hat
            logp = torch.log(p_hat.clamp_min(self.eps))  # (B,C)

            # scores_j = sum_i (alpha_{i,j}-1) * logp_i  (Eq. 20 with w=alpha-1) :contentReference[oaicite:6]{index=6}
            # alpha is (B,i,j) and logp is (B,i) -> (B,j)
            linear_term = torch.einsum("bij,bi->bj", alpha_minus_1, logp)  # (B,C) 
                                   
            if self.quadratic:
                # FIRST CENTER LOG-PROBS
                logp_centered = logp - logp.mean(dim=1, keepdim=True)  # (B,C)                
                Q = self.Q(indices.long())  # (B,S,C)
                P = self.P(indices.long())  # (B,S,C)
                u = torch.einsum("bsc,bc->bs", Q, logp_centered)  # (B,S)
                phi = u**2  # (B,S)
                quadratic_term = torch.sum(P * (T * phi.unsqueeze(-1)), dim=1)  # (B,C)
                linear_term = linear_term + quadratic_term # (B,C)
                
            if self.learn_bias:
                bias = self.bias_code(indices.long())  # (B,S,C)  
                bias = torch.einsum('bsc,s->bc', bias, self.T_code) # (B,C) 
                b = bias
            
            else:                                
                # b_{j|V} = log pi_{j|V} - log B(alpha^{(j)})
                # log B(a) = sum_i lgamma(a_i) - lgamma(sum_i a_i)
                neg_logB = -log_multivariate_beta(alpha, dim=1)
                # alpha_cols = alpha  # (B,C,C), column j = alpha[:, :, j]
                # sum_lgamma = torch.lgamma(alpha_cols).sum(dim=1)          # (B,C)
                # lgamma_sum = torch.lgamma(alpha_cols.sum(dim=1))          # (B,C)
                # logB = sum_lgamma - lgamma_sum                            # (B,C)
                # neg_logB = -logB                                          # (B,C)
                
                if hasattr(self, "pi_logits"):
                    log_pi = F.log_softmax(self.pi_logits, dim=0).unsqueeze(0).expand(B, -1)  # (B,C)
                else:
                    log_pi = torch.log(self.pi.clamp_min(self.eps)).unsqueeze(0).expand(B, -1)  # (B,C)
                                            
                b = log_pi + neg_logB #torch.einsum('bsc,s->bc', bias, self.T_code) # (B,C)                                    

            log_scores = b + linear_term 
            calibrated_probs = F.softmax(log_scores, dim=1)

            return calibrated_probs, log_scores, alpha    

    @torch.no_grad()
    def init_alpha_minus_1_all_1s(self, alpha0: float = 1.0, noise_std: float = 1e-5):
        target_A = softplus_inv(alpha0)          # softplus^{-1}(2) ~= 1.8546
        C, S = self.C, self.S

        # v_i v_j = 1/C for all i,j  -> makes A constant across i,j
        v = torch.ones(C, device=self.A_code.weight.device, dtype=self.A_code.weight.dtype)
        v = v / v.norm()                         # = 1/sqrt(C)

        scale = math.sqrt(target_A * C / max(S, 1))
        base = scale * v                         # (C,)

        self.A_code.weight.copy_(base.unsqueeze(0).repeat(self.K, 1))
        self.B_code.weight.copy_(base.unsqueeze(0).repeat(self.K, 1))

        if noise_std > 0:
            self.A_code.weight.add_(noise_std * torch.randn_like(self.A_code.weight))
            self.B_code.weight.add_(noise_std * torch.randn_like(self.B_code.weight))
            
    @torch.no_grad()
    def init_pi_to_cancel_logB(self):
        # Build a "typical" alpha^{(j)} at init by using average codewords.
        # This is exact only if alpha doesn't depend on x at init (see Option B),
        # but it’s still a very good stabilizer.

        # If you initialized all codewords identically, this is exact.
        W0 = self.A_code.weight.mean(dim=0, keepdim=True)  # (1,C)
        U0 = self.B_code.weight.mean(dim=0, keepdim=True)  # (1,C)

        # Create an (S,C) stack of the same vectors, so A mimics W^T U with S slots
        S = getattr(self, "S", 1)
        W = W0.expand(S, -1).unsqueeze(0)  # (1,S,C)
        U = U0.expand(S, -1).unsqueeze(0)  # (1,S,C)

        A = torch.matmul(W.transpose(1, 2), U)             # (1,C,C)
        alpha = F.softplus(A) + 1e-8                       # (1,C,C)

        sum_lgamma = torch.lgamma(alpha).sum(dim=1)        # (1,C)
        lgamma_sum = torch.lgamma(alpha.sum(dim=1))        # (1,C)
        logB = (sum_lgamma - lgamma_sum).squeeze(0)        # (C,)

        # Make b = log_pi - logB = 0  => log_pi = logB
        self.pi_logits.data.copy_(logB)

    @torch.no_grad()
    def init_bias_to_cancel_logB(self):
        # Build a "typical" alpha^{(j)} at init by using average codewords.
        # This is exact only if alpha doesn't depend on x at init (see Option B),
        # but it’s still a very good stabilizer.

        # If you initialized all codewords identically, this is exact.
        W0 = self.A_code.weight.mean(dim=0, keepdim=True)  # (1,C)
        U0 = self.B_code.weight.mean(dim=0, keepdim=True)  # (1,C)

        # Create an (S,C) stack of the same vectors, so A mimics W^T U with S slots
        S = getattr(self, "S", 1)
        W = W0.expand(S, -1).unsqueeze(0)  # (1,S,C)
        U = U0.expand(S, -1).unsqueeze(0)  # (1,S,C)

        A = torch.matmul(W.transpose(1, 2), U)             # (1,C,C)
        alpha = F.softplus(A) + 1e-8                       # (1,C,C)

        sum_lgamma = torch.lgamma(alpha).sum(dim=1)        # (1,C)
        lgamma_sum = torch.lgamma(alpha.sum(dim=1))        # (1,C)
        logB = (sum_lgamma - lgamma_sum).squeeze(0)        # (C,)

        # Make b = log_pi - logB = 0  => log_pi = logB
        self.bias_code.weight.copy_(logB)
    
    