import torch
from torch import nn
from Model import Model
from torchvision import models

class CNNModel(Model):
    def __init__(self, out_features, lr):
        super().__init__()
        self.network = models.resnet34(pretrained=True)

        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, out_features)
        
        self.lr = lr
        self.criterion = torch.nn.CrossEntropyLoss()
    
    def freeze(self):
        # To freeze the residual layers
        for param in self.network.parameters():
            param.require_grad = False
        for param in self.network.fc.parameters():
            param.require_grad = True
    
    def unfreeze(self):
        # Unfreeze all layers
        for param in self.network.parameters():
            param.require_grad = True

    def accuracy(self, outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))

    def monitor_metrics(self, outputs, targets):
        if targets is None:
            return {}
        outputs = outputs.cpu().detach()
        targets = targets.cpu().detach()
        acc = self.accuracy(outputs, targets)
        return {"Accuracy": acc}

    def fetch_optimizer(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        return opt

    def fetch_scheduler(self):
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1
        )
        return {"sch": sch, "after": "batch", "metric": None}

    def forward(self, x, trg=None):
        out = self.network(x)
        if trg is not None:
            loss = self.criterion(out, trg)
            metrics = self.monitor_metrics(out, trg)
            return out, loss, metrics
        return out, None, None