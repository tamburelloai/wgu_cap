import os
from typing import Dict
import numpy as np
import pandas as pd
import torch
from wwo_hist import retrieve_hist_data
from torch_ds import TorchDS


class DataManager:
    def __init__(self):
        self.cities = self.getCities()
        self.meanValues, self.sigmaValues = self._getAllMeansAndStds()

    def getCities(self) -> Dict:
        '''returns list of city names read from cities directory'''
        cities = {}
        #df = pd.read_csv('data_utils/city_indices.csv')
        df = pd.read_csv('./city_indices.csv')
        for i in range(len(df)):
            cities[df['City'][i]] = i
        return cities

    def getCityFromLabel(self, label):
        for city in self.cities:
            if self.cities[city] == label:
                return city

    def getLabelFromCity(self, city):
        return self.cities[city]


    def getDataset(self, city, split) -> torch.utils.data.Dataset:
        assert split in ['train', 'val', 'test']
        X = self.getData(city, split)
        return TorchDS(X)

    def getFullDataset(self, city):
        def getDate(df):
            return df.year.astype('int').astype('str') + '-' + \
                             df.month.astype('int').astype('str') + '-' + \
                             df.day.astype('int').astype('str')
        def getTime(df):
            return df.hour.astype('int').astype('str') + ':00:00'

        def getDataframe(split):
            return pd.read_csv(f'{split}_data/{city}.csv')
        city = city.replace(' ', '_').lower()
        train_data = getDataframe('train')
        val_data = getDataframe('val')
        test_data = getDataframe('test')
        full_df = pd.concat([train_data, val_data, test_data], axis=0)
        full_df['tempF'] = full_df['tempC'] * 9/5 + 32
        full_df['tempF'] = full_df['tempF'].astype('int')
        full_df['datetime'] = getDate(full_df) + ' ' +  getTime(full_df)
        full_df['datetime'] = pd.to_datetime(full_df['datetime'])
        full_df.sort_values(by='datetime', inplace=True)
        return full_df


    def getData(self, city, split):
        '''get data partition for city'''
        def normalize_data(X):
            mu = self.meanValues[city].reshape(1, -1)
            std = self.sigmaValues[city].reshape(1, -1)
            X -= mu
            X /= (std + 0.00001)
            return X

        X = pd.read_csv(f'{split}_data/{city}.csv')['tempC'].values
        #X = normalize_data(X)
        return X



    def createDatasets(self, city):
        def get_location_data(city):
            frequency = 1
            start_date =  '01-JAN-2018'
            end_date   =  '01-JAN-2019'
            api_key = 'fa745f0d1b7b4fafacd203557221312'
            location_list = [city]
            df = retrieve_hist_data(api_key, location_list, start_date, end_date,
                                    frequency, location_label=False, export_csv=False, store_df=True)[0]
            return df

        def clean_dataset(df):
            df['location'] = self.getLabelFromCity(city)
            df[['date', 'time']] = df['date_time'].astype('str').str.split(' ', expand=True)
            df[['year', 'month', 'day']] = df['date'].str.split('-', expand=True)
            df['hour'] = df['time'].str.split(':', expand=True)[0]
            df = df[['maxtempC', 'mintempC','uvIndex', 'DewPointC', 'cloudcover', 'humidity', 'precipMM',
                     'visibility', 'windspeedKmph', 'location', 'year', 'month', 'day', 'hour', 'tempC']]
            for col in df.columns:
                if col == 'day':
                    df[col] = df[col].astype('int')
                else:
                    df[col] = df[col].astype('float')
            return df

        def getValTestDates():
            val_dates = [np.random.randint(1, 29) for i in range(6)]
            test_dates = [val_dates.pop(np.random.randint(0, len(val_dates))) for i in range(3)]
            return val_dates, test_dates

        def splitDataset(df):
            val_dates, test_dates = getValTestDates()
            val_data = df.loc[df['day'].isin(val_dates)]
            test_data = df.loc[df['day'].isin(test_dates)]
            train_data = df[~df['day'].isin(val_dates + test_dates)]
            return train_data, val_data, test_data

        df = get_location_data(city)
        df = clean_dataset(df)
        train, val, test = splitDataset(df)
        train.to_csv(f'../train_data/{city}.csv', index=False)
        val.to_csv(f'../val_data/{city}.csv', index=False)
        test.to_csv(f'../test_data/{city}.csv', index=False)

    def _getAllMeansAndStds(self):
        means = {}
        stds = {}
        for city in self.cities:
            #df = pd.read_csv(f'train_data/{city}.csv')
            df = pd.read_csv(f'train_data/{city}.csv')
            X = df.values
            mean_values = np.mean(X, axis=0)
            std_values = np.std(X, axis=0)
            means[city] = mean_values
            stds[city] = std_values
        return means, stds

    def getHistorical(self, df):
        return df['tempC'].values

    def getLastWeek(self, city):
        city = city.replace(' ', '_').lower()
        frequency = 1
        start_date = '08-DEC-2022'
        end_date = '15-DEC-2022'
        api_key = 'fa745f0d1b7b4fafacd203557221312'
        location_list = [city]
        df = retrieve_hist_data(api_key, location_list, start_date, end_date,
                                frequency, location_label=False, export_csv=False, store_df=True)[0]
        return df['tempC'].astype('float').values

    def toF(self, data):
        data = np.array(data)
        data *= (9/5)
        data += 32
        return data







def _getAllDatasets():
    dm = DataManager()
    retrieved_cities = []
    for city in os.listdir('train_data'):
        city = city[:-4]
        retrieved_cities.append(city)

    for city in dm.cities:
        if city not in retrieved_cities:
            dm.createDatasets(city)

