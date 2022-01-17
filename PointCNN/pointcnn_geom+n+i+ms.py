import pickle
from myutils import load_dataset, call_home, CMDisplay
from itertools import chain

import torch
import torch.nn.functional as F
from torch.nn import Linear as Lin
from torch.optim import Adam

from torch_geometric.nn import XConv, fps, global_mean_pool

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from sklearn.metrics import classification_report as ClRp
import sklearn.metrics as metrics
import matplotlib.pyplot as plt


class PointCNN(pl.LightningModule):
    def __init__(self, numfeatures=4):
        super().__init__()

        self.learning_rate = 1e-3
        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()
        self.numfeatures = numfeatures
        # First XConv layer.
        # Lifting the point coordinates with no features (0) into feature space
        self.conv1 = XConv(self.numfeatures, 48, dim=3,
                           kernel_size=8, hidden_channels=32)
        # Further XConv layers to further enrich the features
        self.conv2 = XConv(48, 96, dim=3, kernel_size=12,
                           hidden_channels=64, dilation=2)
        self.conv3 = XConv(96, 192, dim=3, kernel_size=16,
                           hidden_channels=128, dilation=2)
        self.conv4 = XConv(192, 384, dim=3, kernel_size=16,
                           hidden_channels=256, dilation=2)

        # MLPs at the end of the PointCNN
        self.lin1 = Lin(389, 256)
        self.lin2 = Lin(256, 128)
        self.lin3 = Lin(128, 4)

    def forward(self, data):
        pos, batch = data.pos, data.batch
        x = data.x if self.numfeatures else None
        ms_feat = data.feat
        # First XConv with no features
        x = F.relu(self.conv1(x, pos, batch))

        # Farthest point sampling, keeping only 37.5%
        idx = fps(pos, batch, ratio=0.375)
        x, pos, batch = x[idx], pos[idx], batch[idx]

        # Second XConv
        x = F.relu(self.conv2(x, pos, batch))

        # Farthest point sampling, keeping only 33.4%
        idx = fps(pos, batch, ratio=0.334)
        x, pos, batch = x[idx], pos[idx], batch[idx]

        # Two more XConvs
        x = F.relu(self.conv3(x, pos, batch))
        x = F.relu(self.conv4(x, pos, batch))

        # Pool the batch-elements together
        # Each tree is described in one single point with 384 features
        x = global_mean_pool(x, batch)

        x = torch.cat((x, ms_feat), dim=1)
        # MLPs at the end with ReLU
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))

        # Dropout (?!): Set randomly values to zero
        x = F.dropout(x, p=0.5, training=self.training)
        # Last MLP predicting labels
        x = self.lin3(x)

        # log-SoftMax Activation function to then calculate NLL-Loss (Negative Log Likelihood)
        return F.log_softmax(x, dim=-1)

    def training_step(self, data, batch_idx):
        y = data.y
        out = self(data)
        loss = F.nll_loss(out, y)
        self.train_acc(out, y)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)
        self.log('train_loss', loss)  # , on_step=True, on_epoch=True)
        return loss

    def validation_step(self, data, batch_idx):
        y = data.y
        out = self(data)
        val_loss = F.nll_loss(out, y)
        self.val_acc(out, y)
        self.log('val_acc', self.val_acc, on_step=True, on_epoch=True)
        self.log('val_loss', val_loss)  # , on_step=True, on_epoch=True)
        return val_loss

    def test_step(self, data, batch_idx):
        y = data.y
        out = self(data)
        test_loss = F.nll_loss(out, y)
        self.test_acc(out, y)
        self.log('test_loss', test_loss)
        return out

    def test_step_end(self, outs):
        return outs

    def test_epoch_end(self, outs):
        global res
        res = outs
        return outs

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


MODEL_NAME = 'geom+n+i+ms'


train_dataset = load_dataset(
    './data/train.h5', batch_size=16, shuffle=True, load_ms=True)
val_dataset = load_dataset(
    './data/val.h5', batch_size=16, load_ms=True)
test_dataset = load_dataset(
    './data/test.h5', batch_size=16, load_ms=True)


checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_top_k=1)
trainer = pl.Trainer(gpus=1,
                     progress_bar_refresh_rate=1,
                     callbacks=[EarlyStopping(
                         monitor='val_loss', patience=20)],
                     checkpoint_callback=checkpoint_callback)

# pl.seed_everything(420)
model = PointCNN()
trainer.fit(model, train_dataset, val_dataset)
best_model = checkpoint_callback.best_model_path
print(best_model)
call_home(f'Done learning {MODEL_NAME}: ' + best_model)

res = []

model = PointCNN.load_from_checkpoint(checkpoint_path=best_model)
# pl.seed_everything(420)
trainer.test(model, test_dataloaders=test_dataset)

with open(f'./results/{MODEL_NAME}_results.pickle', 'wb') as file:
    pickle.dump(res, file)

logits = list(chain(*(r.exp().argmax(axis=1).tolist() for r in res)))
ground = list(chain(*(tmp.y.tolist() for tmp in test_dataset)))

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
