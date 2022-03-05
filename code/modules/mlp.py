from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torchmetrics import Accuracy
from pytorch_lightning.callbacks.progress import base as progress_base


class MlpClassifier(pl.LightningModule):

    def __init__(self, hparams):
        super(MlpClassifier, self).__init__()
        self.save_hyperparameters(hparams)
        layers = [
            # TODO: define model layers here
            # Input self.hparams.num_features
            # Output self.hparams.num_classes
            nn.Linear(self.hparams.num_features, 2048, bias=False),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(2048, self.hparams.num_classes)
        ]
        self.model = nn.Sequential(*layers)
        self.loss = nn.CrossEntropyLoss()
        self.accuracy = Accuracy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        acc = self.accuracy(y_hat, y)
        self.log_dict({'val_loss': loss, 'val_acc': acc},
                      on_epoch=True, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        y_hat = self(x)
        pred = y_hat.argmax(dim=-1)
        return pred

    def configure_optimizers(self):
        # TODO: define optimizer and optionally learning rate scheduler
        # The simplest form would be `return torch.optim.Adam(...)`
        # For more advanced usages, see https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.learning_rate)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                   T_max=self.hparams.scheduler_T_max,
                                                   eta_min=self.hparams.scheduler_eta_min)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler, 
                "interval": "epoch",
            },
        }

    @classmethod
    def add_argparse_args(cls, parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_features', type=int)
        parser.add_argument('--num_classes', type=int, default=15)
        parser.add_argument('--learning_rate', type=float, default=0.01)
        parser.add_argument('--scheduler_T_max', type=int, default=500)
        parser.add_argument('--scheduler_eta_min', type=float, default=1e-6)
        return parser
