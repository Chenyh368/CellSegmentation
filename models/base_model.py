from abc import ABC, abstractmethod
import torch
from torch import nn

class BaseModel(nn.Module):
    def __init__(self, opt, manager, dataloader):
        super(BaseModel, self).__init__()
        self.opt = opt
        self.manager = manager
        self.logger = manager.get_logger()

        # Setup, only support single cuda device
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            raise EnvironmentError()

        self.dataloader = dataloader

        if opt.model.use_amp:
            self.use_amp = True
            self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        else:
            self.use_amp = False
            self.scaler = None
        self.use_sam = self.opt.model.use_sam

    def get_optimizer(self):
        raise NotImplementedError

    def get_schedulers(self):
        if self.opt.model.schedulers is None:
            return dict()
        else:
            raise NotImplementedError()

    def get_criterions(self):
        if self.opt.model.criterions is None:
            return dict()
        else:
            raise NotImplementedError()

    @abstractmethod
    def forward(self, batch):
        """Forward through the network
        """
        raise NotImplementedError()

    def train_step(self, batch):
        """Train a step with a batch of data

        Args:
            batch (dict): dict with keys 'x' (batch, len)
                                         'y' (batch, len)
        """
        self.train()
        x = batch['x'].cuda()
        y = batch['y'].cuda()

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            p = self.forward(x) # forward through the network
            loss, losses = self.calculate_loss(y, p)

        self.optimize(loss)

        return losses, p

    def validation_step(self, batch):
        """Predicts a single batch of data
        Args:
            batch (dict): dict filled with tensors of input and output
        """

        self.eval()
        with torch.no_grad():
            x = batch['x'].to(self.device)
            y = batch['y'].to(self.device)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                p = self.forward(x)  # forward through the network
                _, losses = self.calculate_loss(y, p)

        return losses, p

    def predict_step(self, batch):
        """
        Args:
            batch (dict) dict fill with tensor just for prediction
        """
        self.eval()
        with torch.no_grad():
            x = batch['x'].to(self.device)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                p = self.forward(x)

        return p


    @abstractmethod
    def calculate_loss(self, y, p):
        """Calculates the losses for each criterion

        Args:
            y (tensor): tensor with labels
            p (tensor): tensor with predictions

        Returns:
            loss (tensor): weighted sum of losses
            losses (dict): with detached values for each loss, the weighed sum is named
                global_loss
        """

        raise NotImplementedError()
        return loss, losses

    def optimize(self, loss):
        """Optimizes the model by calculating the loss and doing backpropagation

        Args:
            loss (float): calculated loss that can be backpropagated
        """

        if self.use_sam:
            raise NotImplementedError()
            # TODO
            # it is tricky how to use this SAM thing (https://github.com/davda54/sam)
            # because we have to calculate the loss twice, so we have to find a way
            # to make this general
            # also, it is unclear where to put the gradient clipping

            # loss.backward()
            # self.optimizer.first_step(zero_grad=True)
            # loss, losses = self.calculate_loss(y, p)
            # self.optimizer.second_step(zero_grad=True)
        elif self.scaler is not None:

            self.scaler.scale(loss).backward()

            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.clipping_value)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        else:
            self.optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.parameters(), self.clipping_value)
            self.optimizer.step()

        for scheduler in self.schedulers.values():
            if scheduler:
                scheduler.step()

        return None

    def init_weights(self):
        """Initialize weights from uniform distribution
        """
        for name, param in self.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)

    def count_parameters(self):
        """Count trainable parameters in model
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path):
        """Save the model state
        """
        if self.scaler is not None:
            scaler_dict = self.scaler.state_dict()
        else:
            scaler_dict = None

        save_dict = {'model_state': self.state_dict(),
                     'optimizer_state': self.optimizer.state_dict(),
                     'scaler': scaler_dict}

        for k, v in self.schedulers.items():
            save_dict[k + '_state'] = v.state_dict()
        torch.save(save_dict, path)

    def load(self, path, initialize_lazy = True):
        """Load a model state from a checkpoint file

        Args:
            initialize_lazy (bool): to do a forward step before loading model,
                this solves the problem for layers that are initialized lazyly
        """
        checkpoint_file = path
        if initialize_lazy:
            if self.dummy_batch is None:
                dummy_batch = {'x': torch.randn([16, 1000], device = self.device)}
            else:
                dummy_batch = self.dummy_batch
            self.predict_step(dummy_batch)

        checkpoints = torch.load(checkpoint_file)
        self.load_state_dict(checkpoints['model_state'])

        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoints['optimizer_state'])
        if self.scaler is not None:
            self.optimizer.load_state_dict(checkpoints['scaler'])
        if 'lr_scheduler' in list(self.schedulers.keys()):
            self.schedulers['lr_scheduler'].load_state_dict(checkpoints['lr_scheduler_state'])