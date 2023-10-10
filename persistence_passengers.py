# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 19:36:34 2023

@author: saida
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score


class PersistencePassenger:
    
    
    def __init__(self, filename, nb_lagged):
        
        self.filename = filename
        self.nb_lagged = nb_lagged
        
        
    def structure_data(self):
        
        self.data = pd.read_csv(self.filename, delimiter=',')
        self.data["Month"] = pd.to_datetime(self.data["Month"])
        self.data.set_index("Month", inplace=True)
        
    def analyse(self): 
        
        print(self.data.isnull().sum())
        print('-------------------------------------------------------------')
        print(self.data.describe())
        
    def data_lagged(self):
        
        self.df = self.data.shift(periods=self.nb_lagged)
        self.df.fillna(110, inplace=True)
    
    
    def split_train_test(self):
        
        train_test_size = int(len(self.data)*0.8)
        
        self.train_X = self.data[:train_test_size]
        self.test_X= self.data[train_test_size:]
        
        self.train_y = self.df[:train_test_size]
        self.test_y = self.df[train_test_size:]
        
        return self.train_X, self.test_X, self.train_y, self.test_y
    
    
    def predictions(self):
        
        self.test_predictions = self.test_y
        
        test_mae = mean_absolute_error(self.test_X, self.test_predictions)
        r2 = r2_score(self.test_X, self.test_predictions)
        
        performance = pd.DataFrame({"Metrique ": ["mea", "r2"], "Ensemble de test ":
                                    [test_mae, r2]})
            
        print(performance)
        
    def plot_predictions(self):
        
        plt.figure(figsize=(12, 8))
        
        plt.plot(self.train_X, label='data train')
        plt.plot(self.train_y, label='pred train', c= 'green')
        plt.plot(self.test_X, label='data test', c='red')
        plt.plot(self.test_y, label='pred test', c='black')
        plt.title("Number of passengers 1949-1958")
        plt.xlabel('Date')
        plt.ylabel('Number of passengers')
        plt.legend()



file_name = 'datasets_airline_passengers.csv'
passengers = PersistencePassenger(file_name, nb_lagged=1)
passengers.structure_data()
passengers.analyse()
passengers.data_lagged()
passengers.split_train_test()
passengers.predictions()
passengers.plot_predictions()
