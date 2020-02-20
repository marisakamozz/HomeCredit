import torch
import torch.nn as nn

MAX_LEN = 50


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


class MLPOneHot(nn.Module):
    def __init__(self, n_input, n_hidden):
        super().__init__()
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
        return self.main(x)


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


class R2NOneHotModule(nn.Module):
    def __init__(self, diminfo, n_hidden):
        super().__init__()
        _, _, cont_dim = diminfo
        self.lstm = nn.LSTM(cont_dim, n_hidden)
        self.main = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
        )
    def forward(self, x):
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
        self.onehot = app_cat_dims is None
        if self.onehot:
            app_n_input = app_cont_dim
        else:
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
        if self.onehot:
            self.bureau_layer = R2NOneHotModule(dims['bureau'], n_hidden)
            self.bb_layer = R2NOneHotModule(dims['bureau_balance'], n_hidden)
            self.prev_layer = R2NOneHotModule(dims['previous_application'], n_hidden)
            self.cash_layer = R2NOneHotModule(dims['POS_CASH_balance'], n_hidden)
            self.inst_layer = R2NOneHotModule(dims['installments_payments'], n_hidden)
            self.credit_layer = R2NOneHotModule(dims['credit_card_balance'], n_hidden)
        else:
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
        if self.onehot:
            app, bureau, bb = x[0], x[1], x[2]
            prev, cash, inst, credit = x[3], x[4], x[5], x[6]
            # application
            app = self.app_layer(app)
            # sequences
            bureau = self.bureau_layer(bureau)
            bb = self.bb_layer(bb)
            prev = self.prev_layer(prev)
            cash = self.cash_layer(cash)
            inst = self.inst_layer(inst)
            credit = self.credit_layer(credit)
        else:
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


class R2NCNNModule(nn.Module):
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
        self.conv1 = nn.Conv2d(1, n_hidden, (1, n_input))
        self.conv2 = nn.Conv2d(n_hidden, n_hidden, (MAX_LEN, 1))

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
        x = x.view(x.size(0), 1, x.size(1), x.size(2))
        # x : batch_size * channel(=1) * sequence_length * n_input
        x = torch.relu(self.conv1(x))
        # x : batch_size * n_hidden * sequence_length * 1
        x = torch.relu(self.conv2(x))
        # x : batch_size * n_hidden * 1 * 1
        return x.squeeze(3).squeeze(2)


class R2NCNNOneHotModule(nn.Module):
    def __init__(self, diminfo, n_hidden):
        super().__init__()
        _, _, n_input = diminfo
        self.conv1 = nn.Conv2d(1, n_hidden, (1, n_input))
        self.conv2 = nn.Conv2d(n_hidden, n_hidden, (MAX_LEN, 1))

    def forward(self, x):
        # x : batch_size * sequence_length * n_input
        x = x.view(x.size(0), 1, x.size(1), x.size(2))
        # x : batch_size * channel(=1) * sequence_length * n_input
        x = torch.relu(self.conv1(x))
        # x : batch_size * n_hidden * sequence_length * 1
        x = torch.relu(self.conv2(x))
        # x : batch_size * n_hidden * 1 * 1
        return x.squeeze(3).squeeze(2)


class R2NCNN(nn.Module):
    def __init__(self, dims, n_hidden, n_main):
        super().__init__()
        app_cat_dims, app_emb_dims, app_cont_dim = dims['application_train']
        self.onehot = app_cat_dims is None
        if self.onehot:
            app_n_input = app_cont_dim
        else:
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
        if self.onehot:
            self.bureau_layer = R2NCNNOneHotModule(dims['bureau'], n_hidden)
            self.bb_layer = R2NCNNOneHotModule(dims['bureau_balance'], n_hidden)
            self.prev_layer = R2NCNNOneHotModule(dims['previous_application'], n_hidden)
            self.cash_layer = R2NCNNOneHotModule(dims['POS_CASH_balance'], n_hidden)
            self.inst_layer = R2NCNNOneHotModule(dims['installments_payments'], n_hidden)
            self.credit_layer = R2NCNNOneHotModule(dims['credit_card_balance'], n_hidden)
        else:
            self.bureau_layer = R2NCNNModule(dims['bureau'], n_hidden)
            self.bb_layer = R2NCNNModule(dims['bureau_balance'], n_hidden)
            self.prev_layer = R2NCNNModule(dims['previous_application'], n_hidden)
            self.cash_layer = R2NCNNModule(dims['POS_CASH_balance'], n_hidden)
            self.inst_layer = R2NCNNModule(dims['installments_payments'], n_hidden)
            self.credit_layer = R2NCNNModule(dims['credit_card_balance'], n_hidden)
        self.main = nn.Sequential(
            nn.Linear(n_hidden * 7, n_main),
            nn.BatchNorm1d(n_main),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(n_main, 1),
        )
        
    def forward(self, x):
        if self.onehot:
            app, bureau, bb = x[0], x[1], x[2]
            prev, cash, inst, credit = x[3], x[4], x[5], x[6]
            # application
            app = self.app_layer(app)
            # sequences
            bureau = self.bureau_layer(bureau)
            bb = self.bb_layer(bb)
            prev = self.prev_layer(prev)
            cash = self.cash_layer(cash)
            inst = self.inst_layer(inst)
            credit = self.credit_layer(credit)
        else:
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
        # print(f'app:{app.size()}, bureau:{bureau.size()}, bb:{bb.size()}')
        # print(f'prev:{prev.size()}, cash:{cash.size()}, inst:{inst.size()}, credit:{credit.size()}')
        x = torch.cat([app, bureau, bb, prev, cash, inst, credit], dim=1)
        return self.main(x)


class LSTMEncoder(nn.Module):
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


# for DIM
class GlobalDiscriminator(nn.Module):
    def __init__(self, diminfo, n_hidden):
        super().__init__()
        self.lstm_encoder = LSTMEncoder(diminfo, n_hidden)
        self.main = nn.Sequential(
            # batch_size * n_hidden*2
            nn.Linear(n_hidden * 2, n_hidden),
            nn.ReLU(),
            # batch_size * n_hidden
            nn.Linear(n_hidden, 1)
        )
    
    def forward(self, z, x):
        x = self.lstm_encoder(x)
        # batch_size * n_hidden
        zx = torch.cat([z, x], dim=1)
        # batch_size * n_hidden*2
        return self.main(zx)

# for DIM
class LocalDiscriminator(nn.Module):
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
        self.lstm = nn.LSTM(n_input+n_hidden, n_hidden)
        self.main = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1)
        )
    
    def forward(self, z, x):
        cat, cont = x
        if cat is not None:
            # cat : batch_size * MAX_LEN * number_of_categorical_features
            cat = [emb_layer(cat[:, :, i]) for i, emb_layer in enumerate(self.emb_layers)]
            cat = torch.cat(cat, dim=2)
            # cat : batch_size * MAX_LEN * sum_of_embdims
            x = torch.cat([cat, cont], dim=2)
            # x : batch_size * MAX_LEN * n_input
        else:
            x = cont
        # x : batch_size * MAX_LEN * n_input
        # z : batch_size * n_hidden
        z = z.unsqueeze(-1).transpose(1, 2)
        # z : batch_size * 1 * n_hidden
        z = z.expand(-1, MAX_LEN, -1)
        # z : batch_size * MAX_LEN * n_hidden
        xz = torch.cat([x, z], dim=2)
        # xz : batch_size * MAX_LEN * (n_input+n_hidden)
        xz = xz.transpose(0, 1)
        # xz : MAX_LEN * batch_size * (n_input+n_hidden)
        _, (h, _) = self.lstm(xz)
        # h : 1 * batch_size * n_hidden
        return self.main(h.squeeze(dim=0))

# for DIM
class PriorDiscriminator(nn.Module):
    def __init__(self, n_hidden):
        super().__init__()
        self.main = nn.Sequential(
            # nz
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            # n_hidden
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            # n_hidden
            nn.Linear(n_hidden, 1)
        )
    
    def forward(self, z):
        return self.main(z)


# for VAE
class VAELSTMEncoder(nn.Module):
    def __init__(self, diminfo, n_hidden):
        super().__init__()
        _, _, cont_dim = diminfo
        self.lstm = nn.LSTM(cont_dim, n_hidden)
        self.fc1 = nn.Linear(n_hidden, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)

    def encode(self, x):
        # x : batch_size * sequence_length * n_input
        x = x.transpose(0, 1)
        # x : sequence_length * batch_size * n_input
        _, (h, _) = self.lstm(x)
        # h : 1 * batch_size * n_hidden
        h = h.squeeze(dim=0)
        # h : batch_size * n_hidden
        mu = self.fc1(h)
        logvar = self.fc2(h)
        return mu, logvar
    
    def forward(self, x):
        mu, _ = self.encode(x)
        return mu

# for VAE
class VAELSTM(nn.Module):
    def __init__(self, diminfo, n_hidden):
        super().__init__()
        _, _, cont_dim = diminfo
        self.encoder = VAELSTMEncoder(diminfo, n_hidden)
        self.fc = nn.Linear(n_hidden, cont_dim)
        self.lstm = nn.LSTM(cont_dim, cont_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, x, z):
        # z : batch_size * n_hidden
        z = self.fc(z).unsqueeze(0)
        # z : 1 * batch_size * cont_dim
        c = torch.rand_like(z, device=z.device)
        # x : batch_size * sequence_length * cont_dim
        x = x.transpose(0, 1)
        # x : sequence_length * batch_size * cont_dim
        sos = torch.zeros(1, x.size(1), x.size(2), device=x.device)
        # sos : 1 * batch_size * cont_dim
        x = torch.cat([sos, x], dim=0)[:x.size(0), :, :]
        # x : sequence_length * batch_size * cont_dim
        x, _ = self.lstm(x, (z, c))
        # x : sequence_length * batch_size * cont_dim
        x = x.transpose(0, 1)
        # x : batch_size * sequence_length * cont_dim
        return x

    def forward(self, x):
        mu, logvar = self.encoder.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(x, z), mu, logvar


# for Fine-Tuning
class PretrainedR2N(nn.Module):
    def __init__(self, dims, n_hidden, n_main, encoders):
        super().__init__()
        app_cat_dims, app_emb_dims, app_cont_dim = dims['application_train']
        self.onehot = app_cat_dims is None
        if self.onehot:
            app_n_input = app_cont_dim
        else:
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
        self.bureau_layer = encoders['bureau']
        self.bb_layer = encoders['bureau_balance']
        self.prev_layer = encoders['previous_application']
        self.cash_layer = encoders['POS_CASH_balance']
        self.inst_layer = encoders['installments_payments']
        self.credit_layer = encoders['credit_card_balance']
        self.main = nn.Sequential(
            nn.Linear(n_hidden * 7, n_main),
            nn.BatchNorm1d(n_main),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(n_main, 1),
        )

    def forward(self, x):
        if self.onehot:
            app, bureau, bb = x[0], x[1], x[2]
            prev, cash, inst, credit = x[3], x[4], x[5], x[6]
            # application
            app = self.app_layer(app)
            # sequences
            bureau = self.bureau_layer(bureau)
            bb = self.bb_layer(bb)
            prev = self.prev_layer(prev)
            cash = self.cash_layer(cash)
            inst = self.inst_layer(inst)
            credit = self.credit_layer(credit)
        else:
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
