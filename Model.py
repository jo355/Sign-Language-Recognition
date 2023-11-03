import torch
import torch.nn as nn
from callbacks import CallbackHandler
from tqdm import tqdm

class AverageMonitor:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.valid_loader = None
        self.current_epoch = 0
        self._model_state = None
        self._train_state = None
        self.device = None
        self._cb_Handler = None
        self.metrics = {}
        self.metrics["train"] = {}
        self.metrics["valid"] = {}
        self.metrics["test"] = {}
        self.fp16 = False
        self.scaler = None

    @property
    def model_state(self):
        return self._model_state

    @model_state.setter
    def model_state(self, value):
        self._model_state = value
        
    @property
    def train_state(self):
        return self._train_state

    @train_state.setter
    def train_state(self, value):
        self._train_state = value
        if self._cb_Handler is not None:
            self._cb_Handler(value)


    def _init_model(self, device, callbacks, fp16):

        if callbacks is None:
            callbacks = list()

        self.device = device

        if next(self.parameters()).device != self.device:
            self.to(self.device)

        if self.optimizer is None:
            self.optimizer = self.fetch_optimizer()

        if self.scheduler is None:
            sch_data = self.fetch_scheduler()
            if sch_data is not None:
                self.scheduler = sch_data['sch']
                self.step_scheduler_after = sch_data['after'] if 'after' in sch_data else 'batch'
                self.step_scheduler_metric = sch_data['metric'] if 'metric' in sch_data else None
            else:
                self.scheduler = None

        self.fp16 = fp16
        if self.fp16:
            self.scaler = torch.cuda.amp.GradScaler()

        self._cb_Handler = CallbackHandler(callbacks, self)


    def monitor_metrics(self, *args, **kwargs):
        return
    
    def update_metrics(self, losses, monitor):
        self.metrics[self._model_state].update(monitor)
        self.metrics[self._model_state]["loss"] = losses.avg
    
    def fetch_optimizer(self, *args, **kwargs):
        return

    def fetch_scheduler(self, *args, **kwargs) -> dict:
        return

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

    def model_fn(self, data):
        for key, value in enumerate(data):
            data[key] = value.to(self.device)
        if self.fp16:
            with torch.cuda.amp.autocast():
                output, loss, metrics = self(*data)
        else:
            output, loss, metrics = self(*data)
        return output, loss, metrics

    def scheduler_step(self, after):
        if self.step_scheduler_after == after:
            if self.step_scheduler_metric is None:
                self.scheduler.step()
            else:
                model_state, metric = self.step_scheduler_metric.split("_")
                step_metric = self.metrics[model_state][metric]
                self.scheduler.step(step_metric)

    def train_one_step(self, data):
        self.optimizer.zero_grad()
        self.train_state = 'on_loss_begin'
        _ , loss, metrics = self.model_fn(data)
        loss.backward()
        self.train_state = 'on_step_begin'
        self.optimizer.step()
        
        if self.scheduler:
            self.scheduler_step("batch")

        self.train_state = 'on_step_end'
        return loss, metrics

    def train_one_epoch(self, dataloader):
        self.train()
        self.model_state = 'train'
        losses = AverageMonitor()
        tk0 = tqdm(dataloader, total=len(dataloader))
        for b_idx, data in enumerate(tk0):
            self.train_state = 'on_batch_begin'
            loss, metrics = self.train_one_step(data)
            losses.update(loss.item(), dataloader.batch_size)
            if b_idx == 0:
                metrics_monitor = {k: AverageMonitor() for k in metrics}
            monitor = {}
            for m in metrics_monitor:
                metrics_monitor[m].update(metrics[m], dataloader.batch_size)
                monitor[m] = metrics_monitor[m].avg
            tk0.set_postfix(loss=losses.avg, stage="Train", **monitor)
            self.train_state = 'on_batch_end'
        tk0.close()
        self.update_metrics(losses=losses, monitor=monitor)
        return losses.avg
    
    def validate_one_step(self, data):
        _, loss, metrics = self.model_fn(data)
        return loss, metrics
    
    def validate_one_epoch(self, dataloader):
        self.eval()
        self.model_state = 'valid'
        losses = AverageMonitor()
        tk0 = tqdm(dataloader, total=len(dataloader))
        for b_idx, data in enumerate(tk0):
            with torch.no_grad():
                loss, metrics = self.validate_one_step(data)
            losses.update(loss.item(), dataloader.batch_size)
            if b_idx == 0:
                metrics_monitor = {k: AverageMonitor() for k in metrics}
            monitor = {}
            for m in metrics_monitor:
                metrics_monitor[m].update(metrics[m], dataloader.batch_size)
                monitor[m] = metrics_monitor[m].avg
            tk0.set_postfix(loss=losses.avg, stage="Valid", **monitor)
        tk0.close()
        self.update_metrics(losses=losses, monitor=monitor)
        return losses.avg


    def fit(self, device="cuda", epochs=10, callbacks=None, fp16=False):

        self._init_model(device=device, callbacks=callbacks, fp16=fp16)
        self.device = device
        
        self.train_state = 'on_train_begin'
        for _ in range(epochs):
            self.train_state = 'on_epoch_begin'
            train_loss = self.train_one_epoch(self.train_loader)
            if self.valid_loader:
                valid_loss = self.validate_one_epoch(self.valid_loader)
            
            if self.scheduler:
                self.scheduler_step("epoch")
            
            self.train_state = 'on_epoch_end'
            if self.model_state == "end":
                break
            self.current_epoch += 1
        self.train_state = 'on_train_end'

    def predict_one_step(self, data):
        self.model_state = 'predict'
        self.eval()               
        out, _, _ = self.model_fn(data)
        return out

    def process_output(self, output):
        output = output.cpu().detach().numpy()
        return output

    def predict(self, dataloader, device):
        if next(self.parameters()).device != device:
            self.to(device)

        self.model_state = 'predict'
        self.eval()
        tk0 = tqdm(dataloader, total=len(dataloader))
        final_out = torch.tensor([])
        for data in tk0:
            with torch.no_grad():
                out = self.predict_one_step(data)
                out = out.cpu().detach()
                final_out = torch.cat([final_out, out])
            tk0.set_postfix(stage="Prediction")        
        tk0.close()
        return final_out

    def save(self, model_path):
        model_state_dict = self.state_dict()
        if self.optimizer is not None:
            opt_state_dict = self.optimizer.state_dict()
        else:
            opt_state_dict = None
        if self.scheduler is not None:
            sch_state_dict = self.scheduler.state_dict()
        else:
            sch_state_dict = None
        model_dict = {}
        model_dict["state_dict"] = model_state_dict
        model_dict["optimizer"] = opt_state_dict
        model_dict["scheduler"] = sch_state_dict
        torch.save(model_dict, model_path)
    
    def save_model(self, model_path):
        model_state_dict = self.state_dict()
        model_dict = { "state_dict": model_state_dict }
        torch.save(model_dict, model_path)
    
    def load(self, model_path, device="cpu"):
        if next(self.parameters()).device != device:
            self.to(device)
        model_dict = torch.load(model_path, map_location=torch.device(device))
        self.load_state_dict(model_dict["state_dict"])