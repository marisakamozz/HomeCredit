import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, cat_dims, emb_dims, n_input, n_hidden):
        super().__init__()
        self.emb_layers = nn.ModuleList([
            nn.Embedding(x, y) for x, y in zip(cat_dims, emb_dims)
        ])
        self.main = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(n_hidden, 1),
        )
    def forward(self, x):
        app_cat, app_cont = x[0], x[1]
        app_cat = [emb_layer(app_cat[:, i]) for i, emb_layer in enumerate(self.emb_layers)]
        app_cat = torch.cat(app_cat, dim=1)
        app = torch.cat([app_cat, app_cont], dim=1)
        return self.main(app)


class R2NModule(nn.Module):
    def __init__(self, diminfo, n_hidden):
        super().__init__()
        cat_dims, emb_dims, cont_dim = diminfo
        if cat_dims is not None:
            cat_dims += 1
            self.emb_layers = nn.ModuleList([
                nn.Embedding(x, y) for x, y in zip(cat_dims, emb_dims)
            ])
            n_input = int(emb_dims.sum() + cont_dim)
        else:
            self.emb_layers = None
            n_input = int(cont_dim)
        self.lstm = nn.LSTM(n_input, n_hidden)
        self.main = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
        )
    def forward(self, x):
        cat, cont = x
        if cat is not None:
            # cat : batch_size * sequence_length * number_of_categorical_features
            cat = [emb_layer(cat[:, :, i]) for i, emb_layer in enumerate(self.emb_layers)]
            cat = torch.cat(cat, dim=2)
            # cat : batch_size * sequence_length * sum_of_embdims
            x = torch.cat([cat, cont], dim=2)
            # x : batch_size * sequence_length * n_input
        else:
            x = cont
        # x : batch_size * sequence_length * n_input
        x = x.transpose(0, 1)
        # x : sequence_length * batch_size * n_input
        _, (h, _) = self.lstm(x)
        # h : 1 * batch_size * n_hidden
        return self.main(h.squeeze(dim=0))


class R2N(nn.Module):
    def __init__(self, dims, n_hidden, n_main):
        super().__init__()
        app_cat_dims, app_emb_dims, app_cont_dim = dims['application_train']
        app_n_input = app_emb_dims.sum() + app_cont_dim
        self.app_emb_layers = nn.ModuleList([
            nn.Embedding(x, y) for x, y in zip(app_cat_dims, app_emb_dims)
        ])
        self.app_layer = nn.Sequential(
            nn.Linear(app_n_input, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
        )
        self.bureau_layer = R2NModule(dims['bureau'], n_hidden)
        self.bb_layer = R2NModule(dims['bureau_balance'], n_hidden)
        self.prev_layer = R2NModule(dims['previous_application'], n_hidden)
        self.cash_layer = R2NModule(dims['POS_CASH_balance'], n_hidden)
        self.inst_layer = R2NModule(dims['installments_payments'], n_hidden)
        self.credit_layer = R2NModule(dims['credit_card_balance'], n_hidden)
        self.main = nn.Sequential(
            nn.Linear(n_hidden * 7, n_main),
            nn.BatchNorm1d(n_main),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(n_main, 1),
        )
    def forward(self, x):
        app_cat, app_cont = x[0], x[1]
        bureau_cat, bureau_cont, bb_cat, bb_cont = x[2], x[3], x[4], x[5]
        prev_cat, prev_cont, cash_cat, cash_cont = x[6], x[7], x[8], x[9]
        inst_cat, inst_cont, credit_cat, credit_cont = x[10], x[11], x[12], x[13]
        # application
        app_cat = [emb_layer(app_cat[:, i]) for i, emb_layer in enumerate(self.app_emb_layers)]
        app_cat = torch.cat(app_cat, dim=1)
        app = torch.cat([app_cat, app_cont], dim=1)
        app = self.app_layer(app)
        # sequences
        bureau = self.bureau_layer((bureau_cat, bureau_cont))
        bb = self.bb_layer((bb_cat, bb_cont))
        prev = self.prev_layer((prev_cat, prev_cont))
        cash = self.cash_layer((cash_cat, cash_cont))
        inst = self.inst_layer((inst_cat, inst_cont))
        credit = self.credit_layer((credit_cat, credit_cont))
        # main
        x = torch.cat([app, bureau, bb, prev, cash, inst, credit], dim=1)
        return self.main(x)
