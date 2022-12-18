import os
import torch
from data_utils.data_manager import DataManager
from baseline_model.b_model import LinearRegression

past = 48
future =24
N_EPOCHS = 10000

dm = DataManager()
model = LinearRegression(past, future)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

for epoch in range(N_EPOCHS):
    for i, city in enumerate(dm.cities):
        data = dm.getDataset(city, 'train').data
        for i in range(past, len(data)-(future)):
            X, y = data[i-past:i], data[i:i+future]
            X, y = torch.Tensor(X), torch.Tensor(y)
            yhat = model(X)
            loss = criterion(yhat.view(-1), y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    torch.save(model.state_dict(), 'baseline_state.pt')
