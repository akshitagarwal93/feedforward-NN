import os
import random
from typing import List

import pytest

import numpy as np

import pandas as pd

from keras.layers import Layer, Dropout
from keras.utils import to_categorical
from keras import activations

import tensorflow as tf

import nn


@pytest.fixture(autouse=True)
def set_seeds():
    os.environ['PYTHONHASHSEED'] = '0'
    random.seed(42)
    np.random.seed(42)
    tf.set_random_seed(42)


def test_deep_vs_wide():
    n_in, n_out, train_in, train_out, test_in, test_out = load_auto_mpg()
    deep, wide = nn.create_auto_mpg_deep_and_wide_networks(n_in, n_out)

    # check that the deep neural network is indeed deeper
    assert len(deep.layers) > len(wide.layers)


    # check that the 2 networks have (nearly) the same number of parameters
    params1 = deep.count_params()
    params2 = wide.count_params()
    assert abs(params1 - params2) / (params1 + params2) < 0.05

    # train both networks
    deep.fit(train_in, train_out, verbose=0, epochs=10)
    wide.fit(train_in, train_out, verbose=0, epochs=10)

    # check that error level is acceptable
    [deep_rmse] = root_mean_squared_error(deep.predict(test_in), test_out)
    [wide_rmse] = root_mean_squared_error(wide.predict(test_in), test_out)
    assert deep_rmse < 10
    assert wide_rmse < 10

    # check that the deep neural network is more accurate
    assert deep_rmse < wide_rmse
    print('test passed')
    print(deep_rmse)
    print(wide_rmse)
    print(params1)
    print(params2)
    print(len(deep.layers))


def test_relu_vs_tanh():
    n_in, n_out, train_in, train_out, test_in, test_out = load_delicious(1000)
    relu, tanh = nn.create_delicious_relu_vs_tanh_networks(n_in, n_out)

    # check that models are identical other than the activation functions
    assert len(relu.layers) == len(tanh.layers)
    for relu_layer, tanh_layer in zip(relu.layers, tanh.layers):
        assert relu_layer.__class__ == tanh_layer.__class__
        assert getattr(relu_layer, "units", None) == \
               getattr(tanh_layer, "units", None)

    # check that relu layers are all relu
    relu_activations = [layer.activation
                        for layer in relu.layers[:-1]
                        if hasattr(layer, "activation")]
    assert relu_activations
    assert all(a == activations.relu for a in relu_activations)

    # check that tanh layers are all tanh
    tanh_activations = [layer.activation
                        for layer in tanh.layers[:-1]
                        if hasattr(layer, "activation")]
    assert tanh_activations
    assert all(a == activations.tanh for a in tanh_activations)

    # check that the 2 networks have the same number of parameters
    assert relu.count_params() == tanh.count_params()

    # train both networks
    relu.fit(train_in, train_out, verbose=0, epochs=100)
    tanh.fit(train_in, train_out, verbose=0, epochs=100)

    # check that error levels are acceptable
    relu_accuracy = binary_accuracy(relu.predict(test_in), test_out)
    tanh_accuracy = binary_accuracy(tanh.predict(test_in), test_out)
    all0_accuracy = np.sum(test_out == 0) / test_out.size
    assert relu_accuracy > all0_accuracy
    assert tanh_accuracy > all0_accuracy
    #print("test passed")

def test_dropout():
    n_in, n_out, train_in, train_out, test_in, test_out = load_activity(1000)
    dropout, no_dropout = nn.create_activity_dropout_and_nodropout_networks(
        n_in, n_out)

    # check that the dropout network has Dropout and the other doesn't
    assert any(isinstance(layer, Dropout) for layer in dropout.layers)
    assert all(not isinstance(layer, Dropout) for layer in no_dropout.layers)

    # check that the 2 networks have the same number of parameters
    assert dropout.count_params() == no_dropout.count_params()

    # check that the two networks are identical other than dropout
    dropped_dropout = [l for l in dropout.layers if not isinstance(l, Dropout)]
    assert_layers_equal(dropped_dropout, no_dropout.layers)

    # train both networks
    dropout.fit(train_in, train_out, verbose=0, epochs=10)
    no_dropout.fit(train_in, train_out, verbose=0, epochs=10)

    # check that accuracy level is acceptable
    dropout_accuracy = multi_class_accuracy(dropout.predict(test_in), test_out)
    no_dropout_accuracy = multi_class_accuracy(
        no_dropout.predict(test_in), test_out)
    print("test passed")
    print(dropout_accuracy)
    print(no_dropout_accuracy)
    assert dropout_accuracy >= 0.75
    assert no_dropout_accuracy >= 0.75


    # check that the model with dropout is more accurate
    assert dropout_accuracy > no_dropout_accuracy


def test_early_stopping():
    n_in, n_out, train_in, train_out, test_in, test_out = load_income(2500)
    early, early_fit_kwargs, late, late_fit_kwargs = \
        nn.create_income_earlystopping_and_noearlystopping_networks(n_in, n_out)

    # check that the two networks have the same number of parameters
    assert early.count_params() == late.count_params()

    # check that the two networks have identical layers
    assert_layers_equal(early.layers, late.layers)

    # train both networks
    early_fit_kwargs.update(verbose=0, epochs=100,
                            validation_data=(test_in, test_out))
    early_hist = early.fit(train_in, train_out, **early_fit_kwargs)
    late_fit_kwargs.update(verbose=0, epochs=100)
    late_hist = late.fit(train_in, train_out, **late_fit_kwargs)

    # check that accuracy levels are acceptable
    all0_accuracy = np.mean(test_out == 0)
    early_accuracy = binary_accuracy(early.predict(test_in), test_out)
    late_accuracy = binary_accuracy(late.predict(test_in), test_out)
    print (early_accuracy)
    print (all0_accuracy)
    print (late_accuracy)
    assert early_accuracy > all0_accuracy
    assert late_accuracy > all0_accuracy


    # check that the first network stopped early (fewer epochs)
    print (len(early_hist.history["loss"]))
    print (len(late_hist.history["loss"]))
    assert len(early_hist.history["loss"]) < len(late_hist.history["loss"])

    # check that the first network is more accurate than the second
    assert early_accuracy >= late_accuracy


def load_from_dataframe(df, output_attrs, train_fraction):
    input_df = df.drop(output_attrs, axis=1)
    (_, n_in) = input_df.shape
    output_df = df[output_attrs]
    (_, n_out) = output_df.shape

    mask = np.random.rand(len(df)) < train_fraction
    train_in = input_df[mask].as_matrix()
    train_out = output_df[mask].as_matrix()
    test_in = input_df[~mask].as_matrix()
    test_out = output_df[~mask].as_matrix()
    return n_in, n_out, train_in, train_out, test_in, test_out


def load_auto_mpg():
    df = pd.read_csv("data/Auto-MPG/auto-mpg.data", header=None, sep="\s+",
                     na_values="?",
                     names=["mpg", "cylinders", "displacement", "horsepower",
                            "weight", "acceleration", "model year", "origin",
                            "carname"])
    df = df.dropna().drop("carname", axis=1)
    return load_from_dataframe(df, ["mpg"], 0.8)


def load_delicious(n_rows=None):
    train_in = np.load('data/delicious/delicious-train-features.pkl')
    train_out = np.load('data/delicious/delicious-train-labels.pkl')
    test_in = np.load('data/delicious/delicious-test-features.pkl')
    test_out = np.load('data/delicious/delicious-test-labels.pkl')
    # select just frequent tags
    (tags,) = np.nonzero(np.sum(train_out, axis=0) > 4000)
    train_out = train_out[:, tags]
    test_out = test_out[:, tags]
    (_, n_in) = train_in.shape
    (_, n_out) = train_out.shape
    if n_rows is not None:
        train_in = train_in[:n_rows]
        train_out = train_out[:n_rows]
    return n_in, n_out, train_in, train_out, test_in, test_out


def load_activity(n_rows=None):
    train_in = np.loadtxt("data/UCI-HAR/train/X_train.txt")
    train_out = to_categorical(np.loadtxt("data/UCI-HAR/train/y_train.txt"))
    test_in = np.loadtxt("data/UCI-HAR/test/X_test.txt")
    test_out = to_categorical(np.loadtxt("data/UCI-HAR/test/y_test.txt"))
    (_, n_in) = train_in.shape
    (_, n_out) = train_out.shape
    return n_in, n_out, train_in[:n_rows], train_out[:n_rows], test_in, test_out


def load_income(n_rows=None):
    df = pd.read_csv("data/adult/adult.data", header=None, sep=", ",
                     na_values="?", engine="python", nrows=n_rows, names="""
        age workclass fnlwgt education education-num marital-status occupation
        relationship race sex capital-gain capital-loss hours-per-week
        native-country income""".split())
    df = df.dropna()
    df = pd.get_dummies(df)
    df = df.drop("income_<=50K", axis=1)
    return load_from_dataframe(df, ["income_>50K"], 0.8)


def root_mean_squared_error(system: np.ndarray, human: np.ndarray):
    return ((system - human) ** 2).mean(axis=0) ** 0.5


def multi_class_accuracy(system: np.ndarray, human: np.ndarray):
    return np.mean(np.argmax(system, axis=1) == np.argmax(human, axis=1))


def binary_accuracy(system: np.ndarray, human: np.ndarray):
    return np.mean(np.round(system) == human)


def assert_layers_equal(layers1: List[Layer], layers2: List[Layer]):
    def layer_info(layer):
        return (layer.__class__,
                getattr(layer, "units", None),
                getattr(layer, "activation", None))
    assert [layer_info(l) for l in layers1] == [layer_info(l) for l in layers2]


#test_deep_vs_wide()

#test_relu_vs_tanh()
#test_dropout()
test_early_stopping()



