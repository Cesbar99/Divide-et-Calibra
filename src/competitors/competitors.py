import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from probmetrics.calibrators import get_calibrator
from probmetrics.distributions import CategoricalProbs
import statsmodels.api as sm


class TemperatureScaler(nn.Module):
    def __init__(self, max_iter=50, lr=0.01):
        super(TemperatureScaler, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)  
        self.max_iter = max_iter
        self.lr = lr

    def forward(self, logits):
        return logits / self.temperature

    def predict_proba(self, logits):
        scaled_logits = self.forward(logits)
        return F.softmax(scaled_logits, dim=1)

    def fit(self, val_loader, device="cuda"):
        """
        Optimize temperature using validation set.
        """
        self.to(device)

        logits_list = []
        labels_list = []

        with torch.no_grad():
            for init_feats, init_logits, init_pca, y_one_hot, init_preds, init_preds_one_hot in val_loader:
                init_logits = init_logits.to(device)
                y = y_one_hot.to(device).argmax(dim=1)
                logits_list.append(init_logits)
                labels_list.append(y)

        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)

        # Optimize temperature
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=self.lr, max_iter=self.max_iter)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval)

        print(f"Optimal temperature: {self.temperature.item():.4f}")
        return self

    def calibrated_predictions(self, batch, device="cuda"):
        """Return calibrated probabilities."""        
        with torch.no_grad():
            init_feats, init_logits, init_pca, y_one_hot, init_preds, init_preds_one_hot = batch
            init_logits = init_logits.to(device)                                              
            init_pca = init_pca.to(device)
            init_preds = init_preds.to(device)
            
            new_logits = init_logits / self.temperature
        
            out = {    
                "features": init_pca, 
                "logits": new_logits,        
                "preds": torch.argmax(new_logits, dim=-1).view(-1,1),        
                "true": torch.argmax(y_one_hot, dim=-1).view(-1,1)
            }
        return out        
    

class IsotonicCalibrator():
    """
    Multi-class isotonic regression calibration (one-vs-rest).
    Fits an isotonic regressor per class.
    """
    def __init__(self, out_dim: int):
        self.out_dim = out_dim
        self.models = [IsotonicRegression(out_of_bounds="clip") for _ in range(out_dim)]

    def fit(self, val_loader, device="cuda"):
        logits_list = []
        labels_list = []

        # collect validation data
        with torch.no_grad():
            for init_feats, init_logits, init_pca, y_one_hot, init_preds, init_preds_one_hot in val_loader:
                init_logits = init_logits.to(device)
                y = y_one_hot.to(device).argmax(dim=1)
                logits_list.append(init_logits)
                labels_list.append(y)

        logits = torch.cat(logits_list).cpu().numpy()   
        labels = torch.cat(labels_list).cpu().numpy()   

        # convert to probabilities
        probs = F.softmax(torch.tensor(logits), dim=1).numpy()  

        # one-vs-rest isotonic regression per class
        for c in range(self.out_dim):
            y_binary = (labels == c).astype(float)
            self.models[c].fit(probs[:, c], y_binary)

        return self

    def predict_proba(self, logits, device="cuda"):
        logits = logits.to(device)
        probs = F.softmax(logits, dim=1).cpu().numpy()  
        
        calibrated = np.zeros_like(probs)
        for c, model in enumerate(self.models):
            calibrated[:, c] = model.predict(probs[:, c])

        # renormalize (per sample, so rows sum to 1)
        calibrated /= calibrated.sum(axis=1, keepdims=True)
        return torch.tensor(calibrated, device=device, dtype=torch.float32)

    def calibrated_predictions(self, batch, device="cuda"):
        """Return calibrated probabilities."""        
        with torch.no_grad():
            init_feats, init_logits, init_pca, y_one_hot, init_preds, init_preds_one_hot = batch
            init_logits = init_logits.to(device)                                              
            init_pca = init_pca.to(device)
            init_preds = init_preds.to(device)

            new_logits = self.predict_proba(init_logits)
        
            out = {    
                "features": init_pca, 
                "logits": new_logits,        
                "preds": torch.argmax(new_logits, dim=-1).view(-1,1),        
                "true": torch.argmax(y_one_hot, dim=-1).view(-1,1)
            }
        return out    
    

class PlattScaler():
    """
    Multi-class Platt scaling via one-vs-rest logistic regression.
    """
    def __init__(self, out_dim: int):
        self.out_dim = out_dim
        self.models = [LogisticRegression(solver="lbfgs") for _ in range(out_dim)]

    def fit(self, val_loader, device="cuda"):
        logits_list = []
        labels_list = []

        # collect validation data
        with torch.no_grad():
            for init_feats, init_logits, init_pca, y_one_hot, init_preds, init_preds_one_hot in val_loader:
                init_logits = init_logits.to(device)
                y = y_one_hot.to(device).argmax(dim=1)
                logits_list.append(init_logits)
                labels_list.append(y)

        logits = torch.cat(logits_list).cpu().numpy()   # (N, C)
        labels = torch.cat(labels_list).cpu().numpy()   # (N,)

        # use softmax probs as features
        probs = F.softmax(torch.tensor(logits), dim=1).numpy()  # (N, C)

        # one-vs-rest logistic regression per class
        for c in range(self.out_dim):
            y_binary = (labels == c).astype(int)
            self.models[c].fit(probs, y_binary)

        return self

    def predict_proba(self, logits, device="cuda"):
        logits = logits.to(device)
        probs = F.softmax(logits, dim=1).cpu().numpy()  # (B, C)

        calibrated = np.zeros_like(probs)
        for c, model in enumerate(self.models):
            calibrated[:, c] = model.predict_proba(probs)[:, 1]
        
        calibrated /= calibrated.sum(axis=1, keepdims=True)
        return torch.tensor(calibrated, device=device, dtype=torch.float32)
    
    def calibrated_predictions(self, batch, device="cuda"):              
        with torch.no_grad():
            init_feats, init_logits, init_pca, y_one_hot, init_preds, init_preds_one_hot = batch
            init_logits = init_logits.to(device)                                              
            init_pca = init_pca.to(device)
            init_preds = init_preds.to(device)

            new_logits = self.predict_proba(init_logits)
        
            out = {    
                "features": init_pca, 
                "logits": new_logits,        
                "preds": torch.argmax(new_logits, dim=-1).view(-1,1),        
                "true": torch.argmax(y_one_hot, dim=-1).view(-1,1)
            }
        return out    
    

class DirichletCalibrator(nn.Module):
    def __init__(self, n_classes, lr=0.01, max_iter=100, init_identity=True):
        super().__init__()
        self.n_classes = n_classes                
        self.W = nn.Parameter(torch.zeros(self.n_classes, self.n_classes))
        self.b = nn.Parameter(torch.zeros(self.n_classes))
        if init_identity:
            with torch.no_grad():
                self.W.copy_(torch.eye(n_classes))
        self.lr = lr
        self.max_iter = max_iter

    def _features(self, probs):
        log_probs = torch.log(probs + 1e-12)
        #ones = torch.ones(probs.shape[0], 1, device=probs.device)
        return log_probs #torch.cat([log_probs, ones], dim=1)

    def forward(self, probs):
        features = self._features(probs)  # (N, C+1)
        logits = features @ self.W.t() + self.b  # (N, C)
        return F.softmax(logits, dim=1)

    def fit(self, val_loader, device="cuda"):
        self.to(device)
        probs_list, labels_list = [], []

        # Collect validation set outputs
        with torch.no_grad():
            for batch in val_loader:
                # Adjust depending on your dataloader format
                init_feats, init_logits, init_pca, y_one_hot, init_preds, init_preds_one_hot = batch
                init_logits = init_logits.to(device)
                probs = F.softmax(init_logits, dim=1)
                y = y_one_hot.to(device).argmax(dim=1)

                probs_list.append(probs)
                labels_list.append(y)

        probs = torch.cat(probs_list)     
        labels = torch.cat(labels_list)   

        optimizer = optim.LBFGS([self.W, self.b], lr=self.lr, max_iter=self.max_iter)
        nll_criterion = nn.NLLLoss()

        def closure():
            optimizer.zero_grad()
            q = self.forward(probs)
            loss = nll_criterion(torch.log(q + 1e-12), labels)
            loss.backward()
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            q = self.forward(probs)
            loss = nll_criterion(torch.log(q + 1e-12), labels)
            print(f"Dirichlet calibration training done. Final NLL: {loss.item():.4f}")

        return self

    def calibrated_predictions(self, batch, device="cuda"):
        """Return calibrated probabilities for a batch."""
        with torch.no_grad():
            init_feats, init_logits, init_pca, y_one_hot, init_preds, init_preds_one_hot = batch
            init_logits = init_logits.to(device)
            probs = F.softmax(init_logits, dim=1)
            calibrated_probs = self.forward(probs)
            init_preds = init_preds.to(device)

            return {
                "features": init_pca.to(device),
                "logits": calibrated_probs,
                "preds": torch.argmax(calibrated_probs, dim=-1).view(-1, 1), 
                "true": torch.argmax(y_one_hot, dim=-1).view(-1, 1)
            }
            
                        
class SMS():
    def __init__(self):
        self.sms = get_calibrator("sms")
        
    def fit(self, val_loader, device="cuda"):
        logits_list = []
        labels_list = []
                
        with torch.no_grad():
            for init_feats, init_logits, init_pca, y_one_hot, init_preds, init_preds_one_hot in val_loader:
                init_logits = init_logits.to(device)
                y = y_one_hot.to(device).argmax(dim=1)
                logits_list.append(init_logits.cpu().numpy())
                labels_list.append(y.cpu().numpy())
        logits = np.concatenate(logits_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        probas = F.softmax(torch.tensor(logits), dim=1).numpy()
        
        self.sms.fit(probas, labels)
        print('SMS FITTED!')

    def calibrated_predictions(self, batch, device="cuda"):
        with torch.no_grad():
            init_feats, init_logits, init_pca, y_one_hot, init_preds, init_preds_one_hot = batch
            init_logits = init_logits.to(device)
            probas = F.softmax(init_logits, dim=1).cpu().numpy()
            calibrated_probs = self.sms.predict_proba(probas) #self.sms.predict_proba_torch(CategoricalProbs(probas))
            init_preds = init_preds.to(device)

            return {
                "features": init_pca.to(device),
                "logits": torch.tensor(calibrated_probs, device=device, dtype=torch.float32),
                "preds": torch.argmax(torch.tensor(calibrated_probs), dim=-1).view(-1, 1), 
                "true": torch.argmax(y_one_hot, dim=-1).view(-1, 1)
            }


class DensityRatioCalibration():

    def __init__(self, bandwidth='normal_reference', num_neighbors=10, distance_measure='L2'):
        """
        Params:
            bandwidth:        bandwidth for KDE estimation ('normal_reference' or float)
            num_neighbors:    K for kNN proximity computation
            distance_measure: FAISS index type — 'L2', 'cosine', 'IVFFlat', or 'IVFPQ'
        """
        self.bandwidth = bandwidth
        self.num_neighbors = num_neighbors
        self.distance_measure = distance_measure
        self.faiss_index = None  # built during fit, reused at inference

    def _normalize(self, zs):
        """L2-normalise embeddings row-wise, matching the original paper's preprocessing."""
        return zs / np.linalg.norm(zs, axis=1, keepdims=True)

    def compute_proximities(self, zs, is_val=False):
        """
        Compute scalar kNN proximity scores for a set of embeddings using the
        FAISS index built during fit().

        Embeddings are L2-normalised before querying, bounding all pairwise
        L2 distances to [0, 2] so that exp(-D) never collapses to 0.

        For validation embeddings (is_val=True) the query is its own index, so
        we search K+1 neighbours and drop the self-match (distance = 0).
        For test embeddings (is_val=False) we search K neighbours directly.

        Params:
            zs:     numpy array of embeddings, shape (N, dim)
            is_val: True when querying the same set that was indexed (self-query)

        Returns:
            knndists: numpy array of shape (N,) —
                      mean of exp(-distance) across K neighbours
        """
        assert self.faiss_index is not None, "Call fit() before compute_proximities()."
        zs = self._normalize(zs)
        K = self.num_neighbors
        if is_val:
            D, _ = self.faiss_index.search(zs, K + 1)
            D = D[:, 1:]  # drop self-match
        else:
            D, _ = self.faiss_index.search(zs, K)
        return np.exp(-np.mean(D, axis=1))  # (N,) — exp of mean distance, as per the paper

    def _build_faiss_index(self, val_zs):
        """Build and store the FAISS index from L2-normalised validation embeddings."""
        import faiss
        val_zs = self._normalize(val_zs)
        dim = val_zs.shape[1]
        dm = self.distance_measure

        if dm == "L2":
            index = faiss.IndexFlatL2(dim)
        elif dm == "cosine":
            index = faiss.IndexFlatIP(dim)
        elif dm == "IVFFlat":
            quantizer = faiss.IndexFlatL2(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, 100, faiss.METRIC_L2)
            index.nprobe = 10
            index.train(val_zs)
        elif dm == "IVFPQ":
            quantizer = faiss.IndexFlatL2(dim)
            index = faiss.IndexIVFPQ(quantizer, dim, 100, 8, 8)
            index.nprobe = 10
            index.train(val_zs)
        else:
            raise NotImplementedError(f"Unsupported distance_measure: {dm}")

        index.add(val_zs)
        self.faiss_index = index

    def fit(self, val_loader, val_proximity=None, device="cuda"):
        """
        Train the density estimator on correctly vs misclassified samples.
        1. Collect embeddings (init_feats) and build a FAISS index
        2. Compute val proximities automatically if val_proximity is None
        3. Split samples into correctly classified and misclassified
        4. Learn the conditional KDE of confidence given proximity
           -> distributions are represented as <dens_true, dens_false>

        Params:
            val_loader:    DataLoader yielding
                           (init_feats, init_logits, init_pca, y_one_hot, init_preds, init_preds_one_hot)
                           where init_pca are PCA-reduced embeddings used for proximity
            val_proximity: optional precomputed kNN proximity scores, shape (N,).
                           If None, proximities are computed automatically from init_feats.
            device:        torch device string

        Returns:
            self
        """
        probs_list, preds_list, labels_list, feats_list = [], [], [], []

        with torch.no_grad():
            for init_feats, init_logits, init_pca, y_one_hot, init_preds, init_preds_one_hot in val_loader:
                init_logits = init_logits.to(device)
                probs = F.softmax(init_logits, dim=1)
                y = y_one_hot.to(device).argmax(dim=1)
                preds = init_preds.to(device).view(-1)

                probs_list.append(probs)
                preds_list.append(preds)
                labels_list.append(y)
                feats_list.append(init_pca)  # keep on CPU for FAISS

        probs = torch.cat(probs_list).cpu().numpy()  # (N, C)
        preds = torch.cat(preds_list).cpu().numpy()  # (N,)
        true = torch.cat(labels_list).cpu().numpy()  # (N,)
        val_zs = torch.cat(feats_list).numpy().astype(np.float32)  # (N, dim)

        # Build FAISS index from validation embeddings (always, so predict can reuse it)
        self._build_faiss_index(val_zs)

        # Compute proximities automatically if not provided
        if val_proximity is None:
            print("Computing val proximities from embeddings...")
            proximity = self.compute_proximities(val_zs, is_val=True)
        else:
            proximity = val_proximity  # (N,) — precomputed via np.mean(np.exp(-D), axis=1)

        confs = np.max(probs, axis=-1)

        val_df = pd.DataFrame({'ys': true, 'proximity': proximity, 'conf': confs, 'pred': preds})
        val_df['correct'] = (val_df.pred == val_df.ys).astype('int')

        val_df_true = val_df[val_df['correct'] == 1]
        val_df_false = val_df[val_df['correct'] == 0]

        self.dens_true = sm.nonparametric.KDEMultivariate(
            data=[val_df_true['conf'], val_df_true['proximity']], var_type='cc', bw=self.bandwidth
        )
        self.dens_false = sm.nonparametric.KDEMultivariate(
            data=[val_df_false['conf'], val_df_false['proximity']], var_type='cc', bw=self.bandwidth
        )

        self.false_true_ratio = (val_df.pred != val_df.ys).sum() / (val_df.pred == val_df.ys).sum()

        print('Density Estimation Done.')
        return self

    def predict_proba(self, logits, proximities=None, zs=None, device="cuda"):
        """
        Use Bayes' rule to compute calibrated posterior probabilities:

        p(y_hat=y | h(x), d) = p(h(x), d | y_hat=y) /
            (p(h(x), d | y_hat=y) + p(h(x), d | y_hat!=y) * p(y_hat!=y) / p(y_hat=y))

        Params:
            logits:      torch.Tensor of shape [samples, classes]
            proximities: torch.Tensor or numpy array of shape [samples,], optional.
                         If None, computed automatically from zs.
            zs:          embeddings used to compute proximities, shape [samples, dim],
                         as a torch.Tensor or numpy array. Required if proximities is None.
            device:      torch device string

        Returns:
            calibrated probabilities as torch.Tensor of shape [samples, classes]
        """
        probs = F.softmax(logits.to(device), dim=1).cpu().numpy()  # (N, C)

        if proximities is None:
            assert zs is not None, "Either proximities or zs must be provided."
            if isinstance(zs, torch.Tensor):
                zs = zs.cpu().numpy()
            proximities_np = self.compute_proximities(zs.astype(np.float32), is_val=False)
        elif isinstance(proximities, torch.Tensor):
            proximities_np = proximities.cpu().numpy()
        else:
            proximities_np = proximities  # (N,)

        preds = np.argmax(probs, axis=-1)
        confs = np.max(probs, axis=-1)

        data = np.array([confs, proximities_np]).T  # (N, 2)
        conf_reg_true = self.dens_true.pdf(data)  # (N,)
        conf_reg_false = self.dens_false.pdf(data)

        eps = 1e-10
        conf_calibrated = conf_reg_true / np.maximum(
            conf_reg_true + conf_reg_false * self.false_true_ratio, eps
        )

        # Zero out the predicted class column, redistribute remaining mass
        mask = np.ones(probs.shape, dtype=bool)
        mask[range(probs.shape[0]), preds] = False
        probs = probs * mask
        probs = probs * ((1 - conf_calibrated) / probs.sum(axis=-1))[:, np.newaxis]

        # Insert calibrated confidence for the predicted class
        probs[range(probs.shape[0]), preds] = conf_calibrated

        return torch.tensor(probs, device=device, dtype=torch.float32)

    def calibrated_predictions(self, batch, proximities=None, device="cuda"):
        """
        Return calibrated probabilities for a batch.

        Params:
            batch:       standard dataloader batch tuple
            proximities: optional precomputed kNN proximity scores, shape (B,),
                         as a torch.Tensor or numpy array.
                         If None, proximities are computed automatically from
                         init_feats using the FAISS index built during fit().
            device:      torch device string
        """
        with torch.no_grad():
            init_feats, init_logits, init_pca, y_one_hot, init_preds, init_preds_one_hot = batch
            init_logits = init_logits.to(device)
            init_pca = init_pca.to(device)
            init_preds = init_preds.to(device)

            if proximities is None:
                zs = init_pca.cpu().numpy().astype(np.float32)
                proximities = self.compute_proximities(zs, is_val=False)  # numpy (B,)

            calibrated_probs = self.predict_proba(init_logits, proximities, device=device)

            return {
                "features": init_pca,
                "logits": calibrated_probs,
                "preds": torch.argmax(calibrated_probs, dim=-1).view(-1, 1),
                "true": torch.argmax(y_one_hot, dim=-1).view(-1, 1)
            }