
"""
Copyright © Qibang Liu 2025. All Rights Reserved.

Author: Qibang Liu <qibang@illinois.edu>
National Center for Supercomputing Applications,
University of Illinois at Urbana-Champaign
Created: 2025-01-15
"""

# %%
import torch
import timeit
import os
import json
import numpy as np

# %%


class ModelCheckpoint:
    def __init__(
        self, monitor="val_loss", verbose=0, save_best_only=False, mode="min", filepaths=None
    ):
        self.filepaths = filepaths  # will be updated by the trainer compile if not provided
        self.monitor = monitor
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.best = None
        self.mode = mode

        if self.mode not in ["min", "max"]:
            raise ValueError("mode must be 'min' or 'max'")

        if self.mode == "min":
            self.monitor_op = lambda x, y: x < y
            self.best = float("inf")
        else:
            self.monitor_op = lambda x, y: x > y
            self.best = float("-inf")

    def __call__(self, epoch, logs=None, models=None):
        if len(self.filepaths) != len(models):
            raise ValueError(
                "Number of models is not equal to number of filepaths")

        logs = logs or {}
        current = logs.get(self.monitor)
        current = current[-1]

        if current is None:
            raise ValueError(
                f"Monitor value '{self.monitor}' not found in logs")

        if self.save_best_only:
            if self.monitor_op(current, self.best):
                if self.verbose > 0:
                    print(
                        f"Epoch {epoch + 1}: \
                            {self.monitor} improved from {self.best} to {current}, saving model to {self.filepaths}")

                self.best = current
                for model, filepath in zip(models, self.filepaths):
                    self._save_model(model, filepath)
            else:
                if self.verbose > 1:
                    print(
                        f"Epoch {epoch + 1}: {self.monitor} did not improve from {self.best}")
        else:
            if self.verbose > 0:
                print(f"Epoch {epoch + 1}: saving model to {self.filepaths}")
            for model, filepath in zip(models, self.filepaths):
                self._save_model(model, filepath)

    def _save_model(self, model, filepath):
        torch.save(model.state_dict(), filepath)


class TorchTrainer():
    def __init__(self, models, device, filebase='./saved_models'):
        if isinstance(models, dict):
            self.model_names = list(models.keys())
            models = list(models.values())
        elif isinstance(models, (list, tuple)):
            self.model_names = [f"model{i}" for i in range(len(models))]
            models = models
        else:
            self.model_names = [""]
            models = [models]

        for model in models:
            model.to(device)
        self.models = models
        self.logs = {}
        self.checkpoint = None
        self.device = device
        self.epoch_start = 0
        self.filebase = filebase
        self.window_size = None  # int
        """int list or tuple, indicate the index of sequence in the input data loader"""
        self.sequence_idx = None
        for m_name in self.model_names:
            os.makedirs(os.path.join(filebase, m_name), exist_ok=True)

    def parameters(self):
        combined_params = []
        for model in self.models:
            combined_params += list(model.parameters())
        return combined_params

    def compile(
        self,
        optimizer,
        loss_fn=None,
        checkpoint=None,
        lr_scheduler=None,
        scheduler_metric_name="val_loss",
        window_size=None,
        sequence_idx=None,
    ):
        """
        - This function is used setting up the training environment.
        - Args:
            optimizer: torch.optim.Optimizer
            loss_fn: loss function, may be ovre-written in the evaluate_losses function
            checkpoint: ModelCheckpoint, for saving model weights
            lr_scheduler: torch.optim.lr_scheduler, for learning rate decay
            scheduler_metric_name: str, the metric name for lr_scheduler
            window_size: int, the size of the sliding window
            sequence_idx: int list or tuple, indicate the index of sliding data in the input data loader
        """
        combined_params = []
        for model in self.models:
            combined_params += list(model.parameters())
        self.optimizer = optimizer

        self.checkpoint = checkpoint
        if checkpoint is not None and self.checkpoint.filepaths is None:
            self.checkpoint.filepaths = [os.path.join(
                self.filebase, m_name, 'model.ckpt') for m_name in self.model_names]

        self.loss_fn = loss_fn
        # TODO: Add multiple metrics

        self.lr_scheduler = lr_scheduler
        if lr_scheduler is not None:
            if "lr" not in self.logs:
                self.logs["lr"] = []
            self.metric_name = scheduler_metric_name
        self.window_size = window_size
        self.sequence_idx = sequence_idx

    def collect_logs(self, losses_vals={}):
        for key in losses_vals:
            if key not in self.logs:
                self.logs[key] = []
            self.logs[key].append(
                sum(losses_vals[key]) / len(losses_vals[key]))

    def print_logs(self, epoch, time):
        print(f"Epoch {epoch + 1} took {time:.2f}s", end=", ")
        for key, val in self.logs.items():
            if val:
                print(f"{key}: {val[-1]:.4e}", end=", ")
        print()

    def evaluate_losses(self, data):
        """
        - This loss function is used for both training and validation.
        - if the loss and val_loss are different, need to be re-implemented:
        if self.models[0].training:
            # for training loss
            loss = self.loss_fn(y_pred, y_true)
        else:
            # for validation loss
            loss = self.loss_fn(y_pred, y_true)

        - The current one only works for single model, single input, single output, and simple loss function.
        - Need to be re-implemented using a sub-class if len(self.models) > 1 or more complex loss function.
        args:
            data: torch.utils.data.DataLoader
        output:
            loss: torch.Tensor, for backpropagation
            loss_tracker: dict, for tracking loss values, will be stored in self.logs
        """

        inputs_, y_true = data[0].to(self.device), data[1].to(self.device)
        y_pred = self.models[0](inputs_)
        loss = self.loss_fn(y_pred, y_true)
        loss_tracker = {"loss": loss.item()}
        return loss, loss_tracker

    def train_step(self, data):
        if isinstance(self.optimizer, torch.optim.LBFGS):
            loss_dic = {}

            def closure():
                self.optimizer.zero_grad()
                loss, loss_dic_ = self.evaluate_losses(data)
                loss_dic.update(loss_dic_)
                loss.backward()
                return loss
            self.optimizer.step(closure)
        else:
            self.optimizer.zero_grad()
            loss, loss_dic = self.evaluate_losses(data)
            loss.backward()
            self.optimizer.step()
        return loss_dic

    def validate_step(self, data):
        _, loss_dic = self.evaluate_losses(data)
        val_loss = {}
        for key, value in loss_dic.items():
            val_loss["val_" + key] = value
        return val_loss

    def learning_rate_decay(self, epoch):
        if self.lr_scheduler is None:
            return

        if "lr" not in self.logs:
            self.logs["lr"] = []

        if self.lr_scheduler.__class__.__name__ == "ReduceLROnPlateau":
            if self.metric_name in self.logs:
                self.lr_scheduler.step(self.logs[self.metric_name][-1])
        else:
            self.lr_scheduler.step()

        self.logs["lr"].append(self.lr_scheduler.get_last_lr()[0])

    def slide_window(self, data, loss_vals: dict):
        """
        - This function is used for sliding training for Neural transformer operators.
        - Args:
            data: torch.utils.data.DataLoader,
            self.window_size: int, the size of the sliding window
            self.sequence_idx: int list or tuple, indicate the index of sliding data in the input data loader
                                the data has format of (B,S,...), where B is batch size, S is sequence length
        - Output:
            loss_vals: dict, for tracking loss values, will be stored in self.logs
            no return
        """
        if self.window_size is None or self.sequence_idx is None:
            loss = self.train_step(data)
            for key, value in loss.items():
                if key not in loss_vals:
                    loss_vals[key] = []
                loss_vals[key].append(value)
        else:
            sequence_length = data[self.sequence_idx[0]].shape[1]
            extracted_data = [None]*len(data)
            remain_ids = [i for i in range(
                len(data)) if i not in self.sequence_idx]
            for start in range(0, sequence_length, self.window_size):
                end = min(start + self.window_size, sequence_length)
                extracted_data = [None]*len(data)
                for idx in self.sequence_idx:
                    extracted_data[idx] = data[idx][:, start:end]
                for idx in remain_ids:
                    extracted_data[idx] = data[idx]
                loss = self.train_step(extracted_data)
                for key, value in loss.items():
                    if key not in loss_vals:
                        loss_vals[key] = []
                    loss_vals[key].append(value)

    def fit(
        self,
        train_loader,
        val_loader=None,
        epochs=10,
        print_freq=1,
    ):
        ts = timeit.default_timer()
        loss_vals = {}
        for epoch in range(self.epoch_start, self.epoch_start + epochs):
            # train
            tes = timeit.default_timer()
            for model in self.models:
                model.train()
            loss_vals = {}
            for data in train_loader:
                # loss = self.train_step(data)
                # for key, value in loss.items():
                #     if key not in loss_vals:
                #         loss_vals[key] = []
                #     loss_vals[key].append(value)
                self.slide_window(data, loss_vals)
            self.collect_logs(loss_vals)
            # validate
            if val_loader is not None:
                for model in self.models:
                    model.eval()
                loss_vals = {}
                with torch.no_grad():
                    for data in val_loader:
                        loss = self.validate_step(data)
                        for key, value in loss.items():
                            if key not in loss_vals:
                                loss_vals[key] = []
                            loss_vals[key].append(value)
                self.collect_logs(loss_vals)
            # callbacks at end of epoch
            if self.checkpoint is not None:
                self.checkpoint(epoch, self.logs, self.models)
            # learning rate decay
            self.learning_rate_decay(epoch)

            te = timeit.default_timer()
            if (epoch + 1) % print_freq == 0:
                self.print_logs(epoch, (te - tes))
        print("Total training time:%.2e s" % (te - ts))
        self.epoch_start = epoch + 1
        return self.logs

    def save_logs(self, filebase=None):
        if filebase is None:
            filebase = self.filebase
        if self.logs is not None:
            if not os.path.exists(filebase):
                os.makedirs(filebase, exist_ok=True)
            his_file = os.path.join(filebase, "logs.json")
            with open(his_file, "w") as f:
                json.dump(self.logs, f)

    def load_logs(self, filebase=None):
        if filebase is None:
            filebase = self.filebase
        his_file = os.path.join(filebase, "logs.json")
        if os.path.exists(his_file):
            with open(his_file, "r") as f:
                self.logs = json.load(f)
        return self.logs

    def save_weights(self, filepaths=None):
        if filepaths is None:
            if self.checkpoint is None or self.checkpoint.filepaths is None:
                raise ValueError("No filepaths provided")
            else:
                filepaths = self.checkpoint.filepaths
        elif not isinstance(filepaths, (list, tuple)):
            filepaths = [filepaths]

        if len(filepaths) != len(self.models):
            raise ValueError(
                "Number of models is not equal to number of filepaths")

        for model, path in zip(self.models, filepaths):
            torch.save(model.state_dict(), path)

    def load_weights(self, filepaths=None, device="cpu"):
        if filepaths is None:
            if self.checkpoint is None or self.checkpoint.filepaths is None:
                raise ValueError("No filepaths provided")
            else:
                filepaths = self.checkpoint.filepaths
        elif not isinstance(filepaths, (list, tuple)):
            filepaths = [filepaths]

        if len(filepaths) != len(self.models):
            raise ValueError(
                "Number of models is not equal to number of filepaths")

        for model, path in zip(self.models, filepaths):
            state_dict = torch.load(
                path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()

    def predict(self, data_loader):
        """
        simple Predict on data_loader
        Need to be implemented if len(self.models) > 1
        """
        y_pred = []
        y_true = []
        self.models[0].eval()
        with torch.no_grad():
            for data in data_loader:
                inputs = data[0].to(self.device)
                pred = self.models[0](inputs)
                pred = pred.cpu().detach().numpy()
                y_pred.append(pred)
                y_true.append(data[1].cpu().detach().numpy())
        y_true = np.vstack(y_true)
        y_pred = np.vstack(y_pred)
        return y_pred, y_true

# %%
