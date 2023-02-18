import torch
import pytorch_lightning as pl
from torcheval.metrics import MulticlassAccuracy


class ClsNet(pl.LightningModule):
    def __init__(self, model, opt, loss_fn, num_classes):
        super(ClsNet, self).__init__()

        self.model = model
        self.optimizer = opt
        self.loss_fn = loss_fn
        self.num_classes = num_classes

        # Define metrics
        self.train_acc = MulticlassAccuracy()
        self.val_acc = MulticlassAccuracy()
        self.test_acc = MulticlassAccuracy()


    def forward(self, x) :
        return self.model(x)


    def training_step(self, batch, batch_idx):
        # Define the training step
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)

        # Calculate accuracy
        y_pred = torch.argmax(y_hat, dim=1)
        self.train_acc.update(y_pred.cpu(), y.cpu())

        return {"loss": loss}

    def training_epoch_end(self, outputs):
        # Log accuracy
        acc = self.train_acc.compute()
        self.log('train_acc', acc,prog_bar=True)
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        # Define the validation step
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)

        # Calculate accuracy
        y_pred = torch.argmax(y_hat, dim=1)
        self.val_acc.update(y_pred.cpu(), y.cpu())

        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        # Log accuracy
        acc = self.val_acc.compute()
        self.log('val_acc', acc, prog_bar=True)
        self.val_acc.reset()

    def test_step(self, batch, batch_idx):
        # Define the test step
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)

        # Calculate accuracy
        y_pred = torch.argmax(y_hat, dim=1)
        self.test_acc.update(y_pred.cpu(), y.cpu())

        return {"test_loss": loss}

    def test_epoch_end(self, outputs):
        # Log accuracy
        acc = self.test_acc.compute()
        self.log('test_acc', acc)
        self.test_acc.reset()

    def configure_optimizers(self):
        # Define your optimizer and learning rate scheduler
        optimizer = self.optimizer
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
        return [optimizer], [lr_scheduler]