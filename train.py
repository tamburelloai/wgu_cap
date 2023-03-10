from data_manager import DataManager
from model.transformer import Transformer
from trainer import TorchTrainer

dm = DataManager()

model = Transformer(inpt_features=1,
                    d_model=64,
                    nhead=8,
                    d_hid=64,
                    nlayers=3)


trainer = TorchTrainer(model,
                           batch_size=1,
                           bptt=24,
                           alpha=0.00003,
                           num_epochs=1)

for outer_epoch in range(200):
    train_loss = 0
    val_loss = 0
    for i, city in enumerate(dm.cities):
        # train
        X = dm.getDataset(city, 'train').data
        t_loss = trainer.train_model(X)
        train_loss += t_loss

        # validate
        X = dm.getDataset(city, 'val').data
        v_loss = trainer.evaluate_model(X)
        val_loss += val_loss
        print(f'CITY: {city} | TRAIN LOSS: {t_loss} | VAL LOSS: {v_loss}')

        if i % 5 == 0:
            print('Model Saved')
            trainer.save_model(verified=True)

    trainer.save_model(verified=True)










