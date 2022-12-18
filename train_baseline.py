import os
import torch
from data_manager import DataManager
from b_model import LinearRegression

def train():
    past = 24
    future =24
    N_EPOCHS = 2

    dm = DataManager()
    model = LinearRegression(past)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    for epoch in range(N_EPOCHS):
        for i, city in enumerate(dm.cities):
            data = dm.getDataset(city, 'train').data
            for i in range(past, len(data)-(future)):
                X, y = data[i-past:i], data[i:i+future]
                X, y = torch.Tensor(X), torch.Tensor(y)
                yhat = model(X, future)
                loss = criterion(yhat.view(-1), y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(os.getcwd(), city, loss.item())
            torch.save(model.state_dict(), 'baseline_state.pt')

