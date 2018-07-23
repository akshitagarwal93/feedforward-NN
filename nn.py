from typing import Tuple, Dict

from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout
from keras.models import Model, Sequential
from keras import optimizers
import numpy as np
import pandas as pd


def create_auto_mpg_deep_and_wide_networks(n_inputs: int, n_outputs: int) -> Tuple[Model, Model]:
    deep=Sequential()
    deep.add(Dense(18, activation='sigmoid',input_shape=(n_inputs,)))
    deep.add(Dense(30, activation='sigmoid'))
    deep.add(Dense(50, activation='sigmoid'))
    
    deep.add(Dense(54, activation='sigmoid'))
    deep.add(Dense(58, activation='sigmoid'))
    deep.add(Dense(62, activation='sigmoid'))
    # creating a deep model 

    deep.add(Dense(n_outputs,activation='linear'))
    sgd=optimizers.SGD(lr=0.8,momentum=1.0,decay=1e-6,nesterov=True)
    deep.compile(optimizer='sgd',loss='mean_squared_error',metrics=['accuracy'])

    wide=Sequential()
    wide.add(Dense(100,activation='sigmoid',input_shape=(n_inputs,)))
    wide.add(Dense(100,activation='sigmoid'))
    # creating a wide model 
    wide.add(Dense(n_outputs,activation='linear'))
    wide.compile(optimizer='sgd',loss='mean_squared_error',metrics=['accuracy'])
    return [deep,wide]


def create_delicious_relu_vs_tanh_networks(
        n_inputs: int, n_outputs: int) -> Tuple[Model, Model]:

    relu=Sequential()
    relu.add(Dense(164,activation='relu',input_shape=(n_inputs,)))
    relu.add(Dense(228, activation='relu'))
    relu.add(Dense(328, activation='relu'))
    #relu_model.add(Dense(64, activation='relu'))
    relu.add(Dense(n_outputs,activation='relu'))
    sgd=optimizers.SGD(lr=0.1,momentum=0.9,decay=1e-6,nesterov=True)
    relu.compile(optimizer='sgd',loss='mean_squared_error',metrics=['accuracy'])

    tanh=Sequential()
    tanh.add(Dense(164,activation='tanh',input_shape=(n_inputs,)))
    tanh.add(Dense(228, activation='tanh'))
    tanh.add(Dense(328, activation='tanh'))
    #tanh_model.add(Dense(64, activation='tanh'))
    tanh.add(Dense(n_outputs,activation='tanh'))
    sgd=optimizers.SGD(lr=0.1,momentum=0.9,decay=1e-6,nesterov=True)
    tanh.compile(optimizer='sgd',loss='mean_squared_error',metrics=['accuracy'])
    return [relu,tanh]  
    



def create_activity_dropout_and_nodropout_networks(
        n_inputs: int, n_outputs: int) -> Tuple[Model, Model]:
    """Creates one neural network with dropout applied after each layer, and
    one neural network without dropout. The networks should be identical other
    than the presence or absence of dropout.

    The neural networks will be asked to predict which one of six activity types
    a smartphone user was performing. They will be trained and tested on the
    UCI-HAR dataset from:
    https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones

    :param n_inputs: The number of inputs to the models.
    :param n_outputs: The number of outputs from the models.
    :return: A tuple of (dropout neural network, no-dropout neural network)
    """
    dropout=Sequential()
    dropout.add(Dense(600,activation='relu',input_shape=(n_inputs,)))
    dropout.add(Dropout(0.2))
    dropout.add(Dense(300,activation='relu'))                               
    dropout.add(Dropout(0.2))
   
    dropout.add(Dense(n_outputs,activation='softmax'))
    sgd=optimizers.SGD(lr=0.1,momentum=0.9,decay=1e-16,nesterov=True)
    dropout.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])

    nodrop=Sequential()
    nodrop.add(Dense(600,activation='relu',input_shape=(n_inputs,)))
    nodrop.add(Dense(300,activation='relu'))
    
    nodrop.add(Dense(n_outputs,activation='softmax'))
    sgd=optimizers.SGD(lr=0.009,momentum=0.1,decay=0,nesterov=False)
    nodrop.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])

    return [dropout,nodrop]

     




def create_income_earlystopping_and_noearlystopping_networks(
        n_inputs: int, n_outputs: int) -> Tuple[Model, Dict, Model, Dict]:
    """Creates one neural network that uses early stopping during training, and
    one that does not. The networks should be identical other than the presence
    or absence of early stopping.

    The neural networks will be asked to predict whether a person makes more
    than $50K per year. They will be trained and tested on the "adult" dataset
    from:
    https://archive.ics.uci.edu/ml/datasets/adult

    :param n_inputs: The number of inputs to the models.
    :param n_outputs: The number of outputs from the models.
    :return: A tuple of (
        early-stopping neural network,
        early-stopping parameters that should be passed to Model.fit,
        no-early-stopping neural network,
        no-early-stopping parameters that should be passed to Model.fit
    )
    """

    stop_model=Sequential()
    stop_model.add(Dense(200,activation='tanh',input_shape=(n_inputs,)))
    stop_model.add(Dense(160,activation='tanh'))
    stop_model.add(Dense(120,activation='tanh'))
    stop_model.add(Dense(80,activation='tanh'))
    #stop_model.add(Dense(4,activation='tanh'))
    stop_model.add(Dense(n_outputs,activation='sigmoid'))

    sgd=optimizers.SGD(lr=0.1,momentum=0.9,decay=1e-6,nesterov=True)
    stop_model.compile(optimizer='sgd',loss='binary_crossentropy',metrics=['accuracy'])
    early_stop=EarlyStopping(monitor='loss',min_delta=0.0001,patience=15,verbose=1,mode='auto')
    early_stopping={'verbose':0,'epochs':100,'validation_data':None,'callbacks':[early_stop]}
    late_stopping={'verbose':0,'epochs':10}

    noearly_model=Sequential()
    noearly_model.add(Dense(200,activation='tanh',input_shape=(n_inputs,)))
    noearly_model.add(Dense(160,activation='tanh'))
    noearly_model.add(Dense(120,activation='tanh'))
    noearly_model.add(Dense(80,activation='tanh'))
    #noearly_model.add(Dense(4,activation='tanh'))
    noearly_model.add(Dense(n_outputs,activation='sigmoid'))

    
    noearly_model.compile(optimizer='sgd',loss='binary_crossentropy',metrics=['accuracy'])

    return [stop_model,early_stopping,noearly_model,late_stopping]


    



