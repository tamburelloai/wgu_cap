import math
import torch
from torch import nn, Tensor
from torch.nn import functional as F


class LinearRegression(nn.Module):
    def __init__(self, past):
        super(LinearRegression, self).__init__()
        self.network = nn.Linear(past, 1)
        self.past = past


    def forward(self, x, hours_ahead):
        yhat = []
        for i in range(hours_ahead):
            yhat_i = self.network(x)
            yhat.append(yhat_i.view(-1))
            x = torch.cat([x.view(-1)[1:], yhat_i.view(-1)], dim=-1)
        return torch.stack(yhat, dim=-1)

    def predLastWeek(self, data):
        '''
        next day ahead predictions made by the model over
        given the last 24 hours leading up to the current prediction
        '''
        def remove_prediction_lag(yhat):
            first_pred = [yhat.view(-1)[0]] * 24
            first_pred = torch.stack(first_pred, dim=0)
            yhat = torch.cat([first_pred.view(-1), yhat.view(-1)], dim=-1)
            return yhat

        self.network.eval()
        yhat = []
        with torch.no_grad():
            for i in range(self.past, len(data) - (1)):
                X, y = data[i - self.past:i], data[i:i+1]
                X, y = torch.Tensor(X), torch.Tensor(y)
                yhat_i = self.network(X)
                yhat.append(yhat_i.view(-1))

        yhat = torch.stack(yhat, dim=0)

        #repeat initial pred to make-up for/display lag
        yhat = remove_prediction_lag(yhat)
        return yhat.detach().numpy()

    def predict_historical(self, data):
        '''
        next day ahead predictions made by the model over
        given the last 24 hours leading up to the current prediction
        '''
        def remove_prediction_lag(yhat):
            first_pred = [yhat.view(-1)[0]] * 24
            first_pred = torch.stack(first_pred, dim=0)
            yhat = torch.cat([first_pred.view(-1), yhat.view(-1)], dim=-1)
            return yhat

        self.network.eval()
        yhat = []
        with torch.no_grad():
            for i in range(self.past, len(data) - (1)):
                X, y = data[i - self.past:i], data[i:i+1]
                X, y = torch.Tensor(X), torch.Tensor(y)
                yhat_i = self.network(X)
                yhat.append(yhat_i.view(-1))

        yhat = torch.stack(yhat, dim=0)

        #repeat initial pred to make-up for/display lag
        yhat = remove_prediction_lag(yhat)
        return yhat.detach().numpy()

    def predTomorrow(self, lastWeekActual, days):
        '''
        :param lastWeekActual: last weeks temperatures
        :param days: days head to predict
        :return:
        '''
        yhat = []
        with torch.no_grad():
            lastWeekActual = torch.Tensor(lastWeekActual.reshape(-1)[-24:]) #last 24 hours of data as input
            yhat = self.forward(lastWeekActual, days*24)
        return yhat