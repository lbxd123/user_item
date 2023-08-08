import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score
from Data import DataLoaderMovieLens
from torch.utils.data import DataLoader
# from Data import DataLoaderMl_1M
import torch.nn.functional as F
import torch
from torch import nn
import sys
from sklearn.metrics import roc_curve
from sklearn.metrics import auc, roc_auc_score
torch.set_printoptions(threshold=np.inf)


class CNN(nn.Module):

    def __init__(self, n_user_features, n_item_features, user_df, item_df, dim):
        super(CNN, self).__init__()

        self.user_features = nn.Embedding(n_user_features, dim, max_norm=1)
        self.item_features = nn.Embedding(n_item_features, dim, max_norm=1)

        self.user_df = user_df
        self.item_df = item_df

        total_neighbours = user_df.shape[1] + item_df.shape[1]

        self.Conv1 = nn.Conv1d(in_channels=total_neighbours, out_channels=8, kernel_size=3)
        self.maxpool1 = nn.MaxPool1d(2)
        self.Conv2 = nn.Conv1d(in_channels=8, out_channels=1, kernel_size=3)
        self.maxpool2 = nn.MaxPool1d(2)

        self.dense1 = self.dense_layer(dim // 4 - 2, dim // 8)
        self.dense2 = self.dense_layer(dim // 8, 1)

        self.sigmoid = nn.Sigmoid()

    def dense_layer(self, in_features, out_features):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Tanh()
        )

    def forward(self, u, i, Train=True):
        user_ids = torch.LongTensor(self.user_df.loc[u].values)
        item_ids = torch.LongTensor(self.item_df.loc[i].values)

        user_features = self.user_features(user_ids)
        item_features = self.item_features(item_ids)
        uv = torch.cat([user_features, item_features], dim=1)

        uv = self.Conv1(uv)
        uv = self.maxpool1(uv)
        uv = self.Conv2(uv)
        uv = self.maxpool2(uv)

        uv = torch.squeeze(uv)
        uv = self.dense1(uv)
        uv = F.dropout(uv)
        uv = self.dense2(uv)
        uv = torch.squeeze(uv)

        logit = self.sigmoid(uv)
        return logit


def doEva(net, test_triple):
    d = torch.LongTensor(test_triple)
    u, i, r = d[:, 0], d[:, 1], d[:, 2]
    with torch.no_grad():
        out = net(u, i)
    y_pred = np.array([1 if i >= 0.5 else 0 for i in out])
    auc = roc_auc_score(r, y_pred)
    precision = precision_score(r, y_pred)
    recall = recall_score(r, y_pred)
    acc = accuracy_score(r, y_pred)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, acc, f1, auc



def train(epochs=50, batchSize=512, lr=0.001, dim=128, eva_per_epochs=1, need_eva=True):

    train_triples, test_triples, user_df, item_df, n_user_features, n_item_features = \
        DataLoaderMovieLens.read_data_user_item_df()

    net = CNN(n_user_features, n_item_features, user_df, item_df, dim)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=5e-3)

    for e in range(epochs):
        all_lose = 0
        for u, i, r in DataLoader(train_triples, batch_size=batchSize, shuffle=True):
            r = torch.FloatTensor(r.detach().numpy())
            optimizer.zero_grad()
            logits = net(u, i)
            loss = criterion(logits, r)
            all_lose += loss
            loss.backward()
            optimizer.step()
        print('epoch {},avg_loss={:.4f}'.format(e, all_lose / (len(train_triples) // batchSize)))

        if e % eva_per_epochs == 0 and need_eva:
            p, r, acc, f1, auc = doEva(net, train_triples)
            print('train:p:{:.4f}, r:{:.4f}, acc:{:.4f}, f1:{:.4f} auc:{:.4f}'.format(p, r, acc, f1, auc))

            p, r, acc, f1, auc = doEva(net, test_triples)
            print('train:p:{:.4f}, r:{:.4f}, acc:{:.4f}, f1:{:.4f} auc{:.4f}'.format(p, r, acc, f1, auc))


if __name__ == '__main__':
    train()
