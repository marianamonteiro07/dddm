#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import statsmodels as sm
import statsmodels.tsa.stattools as sm
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import math
from sklearn.metrics import mean_squared_error
import numpy as np


def evaluate(past_moves, past_sales, demand, gain, loss, order_fn):
    """ evaluate student's moves; parameters:
        * past_moves: history of past inventories
        * past_sales: history of past sales
        * demand: true future demand (unknown to students)
        * gain: profit per sold unit
        * loss: deficit generated per unit unsold
        * order_fn: function implementing student's method
    """
    moves = []
    def market(move):
        """ demand function ("censored", as it is limited by 'move'); parameter:
            * move: quantity available for selling (i.e., inventory)
        """
        global nmoves
        if nmoves >= len(demand):
            return None
        moves.append(move)
        sales = min(move, demand[nmoves])
        nmoves += 1
        return sales
    
    profit = 0
    n = len(demand)
    orders = []
    sales = []
    order_fn(past_moves, past_sales, market)

    for i in range(n):
        if moves[i] > demand[i]:
            profit += demand[i]*gain - (moves[i]-demand[i])*loss
        else:
            profit += moves[i]*gain
        print(f"{i+1}\t{demand[i]}\t{moves[i]}\t{moves[i]-demand[i]}\t{profit}")
    return profit


def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return pd.Series(diff)


def timeseries_to_supervised(data, lag=1):
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df


def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()
    return model


def invert_difference(history, y_hat, interval=1):
    return y_hat + history[-interval]


def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, 1)
    y_hat = model.predict(X, batch_size=batch_size, verbose=0)
    return y_hat[0,0]


def order(past_moves, past_sales, market):
    """ function implementing a simple strategy; parameters:
        * past_sales: list with historical data for the market trend
        * market: function to evaluate the actual sales; 'market(value)' returns:
                - inventory, if smaller than demand (everything was sold)
                - true demand otherwise (i.e., some inventory wasn't sold)
    """
    diff_sales = difference(past_sales, 1)
    # transform data to be supervised learning
    supervised = timeseries_to_supervised(diff_sales, 1)
    supervised_values = supervised.values
    train_size = len(past_sales)
    train, test = supervised_values, future_demand

    #training 
    lstm_model = fit_lstm(train, 1, 30, 4)
    sold = past_sales[-1]
    count = 0

    # predicting the test case
    for i in range(len(test)):
        # make one-step forecast
        X = past_sales[-1] - past_sales[-2]
        y_hat_1 = forecast_lstm(lstm_model, 1, X)
        # invert the differencing previously done 
        y_hat = y_hat_1 + sold
        # store forecast
        pred = round(y_hat)
        # report performance
        sold = market(pred)

        np.append(past_sales, sold if sold != pred else sold * 1.025)
        if sold == None:
            return 
        count += 1

if __name__ == "__main__":
    data = pd.read_csv('data-a01.csv', sep='\t')

    past_moves = np.array(data.Inventory[:10000,]) #x_train
    past_sales = np.array([int(i) for i in data.Sales[:10000,]]) #y_train
    future_demand = np.array([int(i) for i in data.Sales[10000:,]]) #x_test

    gain = 1
    loss = 9
    nmoves = 0
    profit = evaluate(past_moves, past_sales, future_demand, gain, loss, order)
    print(profit)
