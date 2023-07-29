import torch
from torch import nn
from layers.scs import SCSTransformer
from models.base_model import BaseModel


class SCSModel(BaseModel):
    def __init__(self, opt, manager, dataloader):
        super(SCSModel, self).__init__(opt, manager, dataloader)
        self._build_network()

        # optimization
        self.optimizer = self.get_optimizer()
        self.schedulers = self.get_schedulers()
        self.criterions = self.get_criterions()

    def _build_network(self):
        o = self.opt.model.arch
        self.model = SCSTransformer(class_num = o.class_num, input_shape = o.input_shape, input_position_shape=o.input_position_shape,
                                    projection_dim=o.projection_dim, num_heads=o.num_heads, transformer_units=o.transformer_units,
                                    transformer_layers=o.transformer_layers, mlp_head_units=o.mlp_head_units)

    def get_optimizer(self):
        if self.opt.model.optimizer is None:
            return None
        elif self.opt.model.optimizer.name == "adamw":
            return torch.optim.AdamW(self.parameters(), lr = self.opt.model.optimizer.lr, weight_decay=self.opt.model.optimizer.weight_decay)
        else:
            raise NotImplementedError()

    def get_criterions(self):
        ce_1 = nn.CrossEntropyLoss(reduction='none')
        ce_2 = nn.BCEWithLogitsLoss()

        return {"pos_ce": ce_1, "cat_ce": ce_2}

    def forward(self, x):
        # Return: pos: [batch_size, num_class] and binary: [batch_size, 1]
        return self.model(x[0], x[1])

    def train_step(self, batch):
        """Train a step with a batch of data

        Args:
            batch (dict): dict with keys 'x': [x_train, x_train_pos]
                                         'y': [y_train, y_binary_train]
        """
        self.train()
        x = [x.cuda() for x in batch['x']]
        y = [y.cuda() for y in batch['y']]

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            p = self.forward(x) # forward through the network
            loss, losses = self.calculate_loss(y, p)

        self.optimize(loss)

        return losses, p

    def calculate_loss(self, y, p):
        # Mask
        loss_pos = torch.mean(self.criterions['pos_ce'](p['pos'], y[0]) * y[1])
        loss_bi = self.criterions['cat_ce'](p['binary'], y[1])

        loss_all = loss_pos + loss_bi
        losses = {'loss.global': loss_all.item(), 'loss.pos': loss_pos.item(), 'loss.bi':loss_bi.item()}

        return loss_all, losses