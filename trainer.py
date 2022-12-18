import os
import time
import torch.nn
from matplotlib import pyplot as plt
from tensor_utils import TensorUtils
from pathlib import Path

plt.ion()

class TorchTrainer:
    def __init__(self, model, batch_size=32, bptt=2, alpha=0.00001,
                 num_epochs=100):
        self.model = model
        self.batch_size = batch_size
        self.alpha = alpha
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.alpha)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.95)
        self.PATH = '../model_state.pt'
        self.bptt = bptt
        self.device = 'cpu'

        self.t_utils = TensorUtils()
        self.num_epochs = num_epochs

    def save_model(self, verified=False):
        assert verified, 'verify action to update model weights'
        torch.save(self.model.state_dict(), self.PATH)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.PATH))

    def train_model(self, train_data):
        total_loss = 0.
        for epoch in range(self.num_epochs):
            total_loss += self._train_model(train_data, epoch)
        return total_loss

    def _train_model(self, train_data, epoch):
        self.model.train()
        log = {'total_loss':0.,
               'log_interval': 50,
               'start_time': time.time()}

        train_data = torch.Tensor(train_data)
        X = self.t_utils.batchify(train_data, self.bptt)
        src_mask = self.t_utils.get_mask(self.bptt).to(self.device)

        num_batches = len(train_data) // self.bptt
        plotGT = []
        plotPRED = []

        for batch, i in enumerate(range(0, train_data.size(0) - 1, self.bptt)):
            data, targets = self.t_utils.get_batch(train_data, i, self.bptt)
            seq_len = data.size(0)
            if seq_len != self.bptt:  # only on last batch
                src_mask = src_mask[:seq_len, :seq_len]
            output = self.model(data, src_mask)
            loss = self.criterion(output.view(-1), targets)
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

            plotGT.extend(targets.data)
            plotPRED.extend(output.view(-1).data)

            log['total_loss'] += loss.item()
            if batch % log['log_interval'] == 0 and batch > 0:
                lr = self.scheduler.get_last_lr()[0]
                ms_per_batch = (time.time() - log['start_time']) * 1000 / log['log_interval']
                cur_loss = log['total_loss'] / log['log_interval']
                #ppl = math.exp(cur_loss)
                #print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                #      f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                #      f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
                total_loss = 0
                start_time = time.time()

        # plt.plot(plotGT)
        # plt.plot(plotPRED)
        # plt.title(f'{epoch}')
        # plt.show()
        # plt.pause(0.1)
        # plt.cla()

        #print(f'| epoch {epoch:3d} | {batch+1:5d}/{num_batches:5d} batches | '
        #      f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
        #      f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
        #print('END OF EPOCH')
        return log['total_loss']

    def evaluate_model(self, eval_data):
        def plot_results(y, yhat):
            plt.plot(y)
            plt.plot(yhat)
            #plt.title(f'{epoch}')
            plt.show()
            plt.pause(0.1)
            plt.cla()

        y = []
        yhat = []
        self.model.eval()
        eval_data = torch.Tensor(eval_data)


        total_loss = 0.
        src_mask = self.t_utils.get_mask(self.bptt).to(self.device)
        with torch.no_grad():
            for i in range(0, eval_data.size(0) - 1, self.bptt):
                data, targets = self.t_utils.get_batch(eval_data, i, self.bptt)
                seq_len = data.size(0)
                if seq_len != self.bptt:
                    src_mask = src_mask[:seq_len, :seq_len]
                output = self.model(data, src_mask).squeeze(-1)
                total_loss += (seq_len * self.criterion(output, targets).item())

                y.extend(targets.data)
                yhat.extend(output.view(-1).data)

        plot_results(y, yhat)
        return total_loss

    def predict_historical(self, X):
        self.model.eval()
        yhat = []
        eval_data = torch.Tensor(X)
        #eval_data = self.model.projection(eval_data).squeeze(-1)

        src_mask = self.t_utils.get_mask(self.bptt).to(self.device)
        with torch.no_grad():
            for i in range(0, eval_data.size(0) - 1, self.bptt):
                data, targets = self.t_utils.get_batch(eval_data, i, self.bptt)
                seq_len = data.size(0)
                if seq_len != self.bptt:
                    src_mask = src_mask[:seq_len, :seq_len]
                output = self.model(data, src_mask).squeeze(-1)
                yhat.extend(output.view(-1).data)

        yhat = torch.stack(yhat, axis=0)
        return yhat.detach().numpy()

    def predLastWeek(self, lastWeek):
        self.model.eval()
        yhat = []
        eval_data = torch.Tensor(lastWeek)
        src_mask = self.t_utils.get_mask(self.bptt).to(self.device)
        with torch.no_grad():
            for i in range(0, eval_data.size(0) - 1, self.bptt):
                data, targets = self.t_utils.get_batch(eval_data, i, self.bptt)
                seq_len = data.size(0)
                if seq_len != self.bptt:
                    src_mask = src_mask[:seq_len, :seq_len]
                output = self.model(data, src_mask).squeeze(-1)
                yhat.extend(output.view(-1).data)

        yhat = torch.stack(yhat, axis=0)
        return yhat.detach().numpy()

    def predTomorrow(self, data, days=1):
        output = []
        yhat = []
        eval_data = torch.Tensor(data)
        src_mask = self.t_utils.get_mask(self.bptt).to(self.device)
        with torch.no_grad():
            for i in range(0, eval_data.size(0) - 1, 1):
                data, targets = self.t_utils.get_batch(eval_data, i, 1)
                seq_len = data.size(0)
                if seq_len != self.bptt:
                    src_mask = src_mask[:seq_len, :seq_len]
                output = self.model(data, src_mask).squeeze(-1)
                pred = output[-1]
                eval_data = eval_data[1:] # remove oldest datapoint
                eval_data = torch.cat([eval_data, pred.view(-1)], axis=0)
                yhat.append(pred)
        return yhat[:24*days]



