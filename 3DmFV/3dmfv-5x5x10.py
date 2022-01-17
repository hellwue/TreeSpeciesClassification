import torch
import torch.nn.functional as F
from torch.nn import Linear as Lin
from torch.optim import Adam
from torch.nn import MaxPool3d
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import matplotlib.pyplot as plt

from myutils import CMDisplay, call_home, Inception

import pickle

from sklearn.metrics import classification_report as ClRp
import sklearn.metrics as metrics


class Net_3DmFV(pl.LightningModule):
    def __init__(
        self,
        train_data_path="./data/train_5x5x10.pickle",
        val_data_path="./data/val_5x5x10.pickle",
        test_data_path="./data/test_5x5x10.pickle",
        batch_size=64,
        lrschedule=True,
    ):
        super().__init__()

        self.lrschedule = lrschedule
        self.batch_size = batch_size
        self.learning_rate = 1e-3
        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()

        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.test_data_path = test_data_path

        self.inception0 = Inception(20, 64, 3, 5)
        self.inception1 = Inception(64 * 3, 128, 3, 5)
        self.inception2 = Inception(128 * 3, 256, 3, 5)
        self.inception3 = Inception(256 * 3, 256, 2, 3)
        self.inception4 = Inception(256 * 3, 512, 2, 3)
        self.maxpool = MaxPool3d(3, 2)

        self.lin0 = Lin(512 * 3 * (2 * 2 * 4), 1024)
        self.lin1 = Lin(1024, 256)
        self.lin2 = Lin(256, 128)
        self.lin3 = Lin(128, 4)

    def train_dataloader(self):
        with open(self.train_data_path, "rb") as file:
            trees = pickle.load(file)
        return DataLoader(trees, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        with open(self.val_data_path, "rb") as file:
            trees = pickle.load(file)
        return DataLoader(trees, batch_size=self.batch_size)

    def test_dataloader(self):
        with open(self.test_data_path, "rb") as file:
            trees = pickle.load(file)
        return DataLoader(trees, batch_size=self.batch_size)

    def forward(self, data):
        # Inceptions with size 5
        data = self.inception0(data)
        data = self.inception1(data)
        data = self.inception2(data)
        # Pooling
        data = self.maxpool(data)
        # Inceptions with size 2
        data = self.inception3(data)
        data = self.inception4(data)

        # FCN:
        data = torch.reshape(data, [data.shape[0], -1])
        data = F.relu(self.lin0(data))
        data = F.dropout(data, 0.3, training=self.training)
        data = F.relu(self.lin1(data))
        data = F.dropout(data, 0.3, training=self.training)
        data = F.relu(self.lin2(data))
        data = F.dropout(data, 0.3, training=self.training)
        data = self.lin3(data)
        return F.log_softmax(data, dim=-1)

    def training_step(self, tree_batch, batch_idx):
        y = tree_batch["label"].reshape(-1)
        data = tree_batch["points"]
        out = self(data)
        loss = F.nll_loss(out, y)
        self.log("train_acc_step", self.train_acc(out, y))
        self.log("train_loss", loss)
        return loss

    def training_epoch_end(self, outs):
        self.log("train_acc_epoch", self.train_acc.compute())

    def validation_step(self, tree_batch, batch_idx):
        y = tree_batch["label"].reshape(-1)
        data = tree_batch["points"]
        out = self(data)
        val_loss = F.nll_loss(out, y)
        self.val_acc(out, y)
        self.log("val_acc", self.val_acc, on_epoch=True)
        self.log("val_loss", val_loss)
        return val_loss

    def test_step(self, tree_batch, batch_idx):
        y = tree_batch["label"].reshape(-1)
        data = tree_batch["points"]
        out = self(data)
        test_loss = F.nll_loss(out, y)
        self.test_acc(out, y)
        self.log("test_acc", self.test_acc, on_step=True, on_epoch=True)
        self.log("test_loss", test_loss)
        global test_y
        global test_out
        test_y.append(y)
        test_out.append(out)
        return out

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        if self.lrschedule:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=20, gamma=0.7
            )
            return [optimizer], [scheduler]
        else:
            return optimizer


pl.seed_everything()


MODEL_NAME = '5x5x10'

model = Net_3DmFV(batch_size=64)

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss", save_top_k=1, save_weights_only=True)
trainer = pl.Trainer(
    gpus=1,
    min_epochs=10,
    progress_bar_refresh_rate=1,
    callbacks=[EarlyStopping(monitor="val_loss", patience=20)],
    checkpoint_callback=checkpoint_callback,
    precision=16,
)

trainer.fit(model)
best_model = checkpoint_callback.best_model_path
print(best_model)
call_home(f'Done learning {MODEL_NAME}: ' + best_model)

test_y = []
test_out = []
model = Net_3DmFV.load_from_checkpoint(checkpoint_path=best_model)
trainer.test(model)


logits = torch.cat([x.max(axis=1).indices for x in test_out]).cpu()
ground = torch.cat(test_y).cpu()

# from itertools import chain
# logits = list(chain(*(r.exp().argmax(axis=1).tolist() for r in res)))
# ground = list(chain(*(tmp.y.tolist() for tmp in test_dataset)))


classification_report = ClRp(ground,
                             logits,
                             target_names=['coniferous',
                                           'decidious',
                                           'snag',
                                           'dead tree'],
                             digits=3)
print(classification_report)
with open(f'./results/{MODEL_NAME}_results.txt', 'w') as file:
    file.writelines(classification_report)
    file.writelines(best_model)


CMDisplay(metrics.confusion_matrix(ground, logits)).plot()
plt.savefig(f'./results/{MODEL_NAME}_results.eps', bbox_inches='tight')
plt.savefig(f'./results/{MODEL_NAME}_results.png', bbox_inches='tight')
