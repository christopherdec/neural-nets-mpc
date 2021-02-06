# -*- coding: utf-8 -*-
"""Módulo de modelos customizados

Contém a implementação em classe de modelos do tensorflow, dentre eles
o DLP, LSTM, GRU e Simple RNN. Os dois últimos foram movidos para
unused_functions.py, por não serem mais utilizados no código.
"""

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Lambda, Concatenate
from tensorflow.keras.losses import Loss


class SSE(Loss):
    """Implementação da função custo Sum of Squared Errors

    Resultados muito semelhantes à utilização do MSE
    """
    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        error = tf.math.subtract(y_true, y_pred)
        squared_error = tf.math.square(error)
        return tf.math.reduce_sum(squared_error, axis=-1)


def serial_parallel_model(K, loss='mse', optimizer='adam'):
    """Implementação do DLP utilizando a Sequential API

    Exclusivo para horizonte = 1 (configuração de trainamento series-parallel)

    Args:
        K (int): número de nodos na hidden layer
        loss (str, Loss, optional): função de custo. Defaults to 'mse'.
        optimizer (str, Optimizer, optional): Defaults to 'adam'.

    Returns:
        Sequential: modelo compilado
    """
    model = Sequential([
        Dense(K, activation='tanh', name='hidden'),
        Dense(1, activation='linear', name='output')
    ])
    model.compile(optimizer=optimizer, loss=loss)
    return model


class ParallelModel(tf.keras.Model):
    """Implementação em classe do DLP utilizando a Functional API

    Para treinamento na configuração parallel (horizonte > 1)

    Args:
            horizon (int): tamanho da janela de predição
            K (int): número de nodos na hidden layer
            loss (str, Loss, optional): função de custo. Defaults to 'mse'.
            optimizer (str, Optimizer, optional): Defaults to 'adam'.
    """

    def __init__(self, horizon, K, loss='mse', optimizer='adam'):
        super().__init__()
        self.horizon = horizon
        self.hidden_layer = tf.keras.layers.Dense(
            K, activation='tanh', name='hidden')
        self.output_layer = tf.keras.layers.Dense(
            1, activation='linear', name='output')
        self.compile(optimizer=optimizer, loss=loss)

    def call(self, inputs):
        """Chamada para realizar as predições

        Recebe o vetor y0, que informa o estado inicial de cada janela, com
        dimensão (ordem de "y", exemplos).

        Também recebe as variáveis de entrada "u", num vetor com dimensão
        (entradas "u", timestep, exemplos)

        Um for percorre cada step do horizonte, e os valores calculados de 'y'
        são concatenados na entrada da rede com os valores dos 'u' para o
        respectivo timestep

        Args:
            inputs (list): lista com o vetor y0 e vetor de inputs

        Returns:
            list: predições
        """
        y_pred = inputs[1]

        u = inputs[0]

        predictions = []

        for t in range(0, self.horizon):

            x = u[:, t, :]

            x = tf.concat([x, y_pred], axis=1)

            x = self.hidden_layer(x)

            prediction = self.output_layer(x)

            predictions.append(prediction)

            y_pred = tf.roll(y_pred, shift=1, axis=1)

            y_pred = y_pred[:, 1:]

            y_pred = tf.concat([prediction, y_pred], axis=1)

        # predictions.shape = (time, batch, features)
        predictions = tf.stack(predictions)

        # predictions.shape => (batch, time, features)
        predictions = tf.transpose(predictions, [1, 0, 2])

        return predictions


class LSTM(tf.keras.Model):
    """Implementação em classe da LSTM utilizando a Functional API

    Para treinamento na configuração parallel (horizonte > 1)

    Baseado no tutorial de Timeseries do tensorflow

    Pode ser utilizada apenas no input selection (alterando um parâmetro do
    exec_cfg na main.py), porém o modelo final deve ser um DLP (sem
    recursividade interna).

    Args:
            horizon (int): tamanho da janela de predição
            K (int): número de nodos na hidden layer
            loss (str, Loss, optional): função de custo. Defaults to 'mse'.
            optimizer (str, Optimizer, optional): Defaults to 'adam'.
    """
    def __init__(self, horizon, K, loss='mse', optimizer='adam'):
        super().__init__()
        self.horizon = horizon
        self.units = K
        self.lstm_cell = tf.keras.layers.LSTMCell(K)
        # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
        # num_features = 1 feature pois a rede é MISO
        self.dense = tf.keras.layers.Dense(1, activation='linear')
        self.compile(optimizer=optimizer, loss=loss)

    def warmup(self, inputs):
        """The first method this model needs is a warmup method to initialize
        its internal state based on the inputs. Once trained this state will
        capture the relevant parts of the input history. This is equivalent to
        the single-step LSTM model from earlier
        """
        y0_orig = inputs[1]

        y0 = tf.expand_dims(y0_orig, axis=1)

        x = inputs[0]

        x = x[:, 0:1, :]

        x = tf.concat([x, y0], axis=2)

        # inputs.shape => (batch, time, features)

        # x.shape => (batch, lstm_units)

        x, *state = self.lstm_rnn(x)

        # predictions.shape => (batch, features)

        prediction = self.dense(x)

        y_pred = tf.roll(y0_orig, shift=1, axis=1)

        y_pred = Lambda(lambda z: z[:, 1:])(y_pred)

        y_pred = Concatenate()([prediction, y_pred])

        return y_pred, prediction, state

    def call(self, inputs, training=None):
        """With the RNN's state, and an initial prediction you can now
        continue iterating the model feeding the predictions at each step
        back as the input. The simplest approach to collecting the output
        predictions is to use a python list, and tf.stack after the loop
        """
        # Initialize the lstm state
        y_pred, prediction, state = self.warmup(inputs)

        inputs = inputs[0]

        # Use a TensorArray to capture dynamically unrolled outputs and insert
        # the first prediction
        predictions = [prediction]

        # Run the rest of the prediction steps. Começa em 1 porque o warmup já
        # fez a primeira predição!
        for n in range(1, self.horizon):

            x = Lambda(lambda z: z[:, n, :])(inputs)

            x = Concatenate()([x, y_pred])

            # Execute one lstm step.
            x, state = self.lstm_cell(x, states=state, training=training)

            # Convert the lstm output to a prediction.
            prediction = self.dense(x)

            # Add the prediction to the output
            predictions.append(prediction)

            y_pred = tf.roll(y_pred, shift=1, axis=1)

            y_pred = Lambda(lambda z: z[:, 1:])(y_pred)

            y_pred = Concatenate()([prediction, y_pred])

        # predictions.shape => (time, batch, features)
        predictions = tf.stack(predictions)

        # predictions.shape => (batch, time, features)
        predictions = tf.transpose(predictions, [1, 0, 2])

        return predictions
