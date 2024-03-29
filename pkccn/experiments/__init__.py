import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from time import time

class MLPClassifier(BaseEstimator, ClassifierMixin):
    """A simple neural network."""
    def __init__(
            self,
            activation = "sigmoid",
            hidden_layer_sizes = (50,),
            alpha = .001,
            max_iter = 200,
            class_weight = None,
            n_repeats = 1,
        ):
        super().__init__()
        self.activation = activation
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.n_repeats = n_repeats
    def fit(self, X, _y):
        self.classes_ = np.unique(_y)
        if len(self.classes_) != 2:
            raise ValueError(f"More than two classes {self.classes_}")
        y = np.ones_like(_y, dtype=int)
        y[_y==self.classes_[0]] = -1 # y in [-1, +1]
        w = np.ones_like(y, dtype=float) # set up sample weights
        if self.class_weight == "balanced":
            w[y==1] = len(y) / np.sum(y==1) / 2
            w[y==-1] = len(y) / np.sum(y==-1) / 2
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        w = torch.tensor(w, dtype=torch.float32).unsqueeze(1)

        # multi-start optimization
        best = (np.inf, None) # (loss, module)
        for i in range(self.n_repeats):
            # start = time()
            current = self._fit_once(X, y, w)
            # print(f"Loss {i}/{self.n_repeats}: {current[0]:.5f} in {time() - start:.2f}s")
            if current[0] < best[0]:
                best = current
        self.module = best[1]
        return self
    def _fit_once(self, X, y, w):
        module = MLPModule(
            X.shape[1],
            self.activation,
            self.hidden_layer_sizes,
        )
        criterion = nn.MSELoss(reduction="none")
        optimizer = optim.LBFGS(
            module.parameters(),
            line_search_fn = "strong_wolfe",
            max_iter = 20, # default: 20
            tolerance_grad = 1e-3, # default: 1e-5
            tolerance_change = 1e-6 # default: 1e-9
        )
        module.train()
        def closure():
            if torch.is_grad_enabled():
                optimizer.zero_grad()
            loss = (w * criterion(module(X), y)).mean()
            if loss.requires_grad:
                for n, p in module.named_parameters():
                    if "weight" in n: # L2 regularization
                        loss += self.alpha * torch.norm(p).pow(2)
                loss.backward()
                # print(f"{i:03d} -> {loss.item():.4e}")
            return loss
        for i in range(self.max_iter): # training
            optimizer.step(closure)
        with torch.no_grad(): # evaluate the loss of this fit
            loss = closure().item()
        return loss, module
    def decision_function(self, X):
        return self.module(torch.tensor(X, dtype=torch.float32)).detach().numpy().flatten()
    def predict_proba(self, X):
        y_pred = (self.decision_function(X) + 1) / 2
        y_pred = np.stack((1-y_pred, y_pred)).T
        return y_pred / y_pred.sum(axis=1, keepdims=True)
    def predict(self, X):
        return (self.decision_function(X) > 0) * 2 - 1

class MLPModule(nn.Module):
    """A simple neural network."""
    def __init__(
            self,
            input_size,
            activation,
            hidden_layer_sizes,
        ):
        super().__init__()
        if activation == "sigmoid":
            activation = nn.Sigmoid
        else:
            raise ValueError(f"activation={activation} is not available")
        layers = [nn.Linear(input_size, hidden_layer_sizes[0])]
        for i in range(1, len(hidden_layer_sizes)):
            layers.extend([
                activation(),
                nn.Linear(hidden_layer_sizes[i-1], hidden_layer_sizes[i])
            ])
        layers.extend([
            activation(),
            nn.Linear(hidden_layer_sizes[-1], 1)
        ])
        self.layers = nn.Sequential(*layers)
    def forward(self, X):
        return self.layers(X)
