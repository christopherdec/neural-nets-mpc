"""Funções não utilizadas atualmente

Contém funções desenvolvidas que não são mais utilizadas no código,
foram trazidas para cá para melhorar a organização. Contém funções
que eram de training_utils.py e custom_models.py.
"""
# Estes imports servem apenas para o Sphinx não 'reclamar' durante o processo
# de documentação
import tensorflow as tf
from tensorflow.keras.layers import Dense, Concatenate, Lambda, GRUCell, RNN, \
    SimpleRNNCell
import time
import pandas as pd
import numpy as np
from . import data_utils as du
from . import custom_models as cm


class SerialParallelModel(tf.keras.Model):
    """Implementação do DLP utilizando a Functional API

    Exclusivo para horizonte = 1 (configuração series-parallel)

    Args:
        K (int): número de nodos na hidden layer
        loss (str, Loss, optional): função de custo. Defaults to 'mse'.
        optimizer (str, Optimizer, optional): Defaults to 'adam'.
    """
    def __init__(self, K, loss='mse', optimizer='adam'):

        super(SerialParallelModel, self).__init__()
        self.hidden_layer = Dense(
            K, activation='tanh', name='hidden')
        self.output_layer = Dense(
            1, activation='linear', name='output')

        self.compile(optimizer=optimizer, loss=loss)

    def call(self, inputs):
        """With the RNN's state, and an initial prediction you can now
        continue iterating the model feeding the predictions at each step
        back as the input. The simplest approach to collecting the output
        predictions is to use a python list, and tf.stack after the loop
        """
        x = self.hidden_layer(inputs)
        return self.output_layer(x)


class ParallelModel2(tf.keras.Model):
    """Implementação em classe do DLP utilizando a Functional API

    Implementado de uma maneira diferente, mas deu os mesmos resultados

    Para treinamento na configuração parallel (horizonte > 1)

    Args:
            regressors (pd.Series): input variables e respectiva ordem
            output(str): saída para a qual a rede está sendo criada
            horizon (int): tamanho da janela de predição
            K (int): número de nodos na hidden layer
            loss (str, Loss, optional): função de custo. Defaults to 'mse'.
            optimizer (str, Optimizer, optional): Defaults to 'adam'.
    """
    def __init__(self,
                 regressors,
                 output,
                 horizon,
                 K,
                 loss='mse',
                 optimizer='adam',
                 **kwargs):
        super().__init__(**kwargs)
        self.hidden_layer = Dense(K, input_shape=(
            sum(regressors),), activation='tanh')
        self.output_layer = Dense(1, activation='linear')
        self.horizon = horizon
        self.y_order = regressors[output]
        self.concat = Concatenate()
        self.compile(optimizer=optimizer, loss=loss)

    def call(self, inputs):
        """With the RNN's state, and an initial prediction you can now
        continue iterating the model feeding the predictions at each step
        back as the input. The simplest approach to collecting the output
        predictions is to use a python list, and tf.stack after the loop
        """

        X, y0 = inputs[0], inputs[1]

        for i in range(self.y_order):
            vars(self)[f'feedback_{i}'] = y0[:, i:i + 1]

        predictions = []

        for t in range(self.horizon):

            x = X[:, t, :]

            for i in range(self.y_order):
                x = self.concat([x, vars(self)[f'feedback_{i}']])

            x = self.hidden_layer(x)

            prediction = self.output_layer(x)

            predictions.append(prediction)

            for i in range(self.y_order - 1, 0, -1):
                vars(self)[f'feedback_{i}'] = vars(self)[f'feedback_{i-1}']

            self.feedback_0 = prediction

        # predictions.shape = (time, batch, features)
        predictions = tf.stack(predictions)

        # predictions.shape => (batch, time, features)
        predictions = tf.transpose(predictions, [1, 0, 2])

        return predictions


class GRU(tf.keras.Model):
    """Implementação em classe da GRU utilizando a Functional API

    Para treinamento na configuração parallel (horizonte > 1)

    Não utilizado e desatualizado.

    Args:
            horizon (int): tamanho da janela de predição
            K (int): número de nodos na hidden layer
    """
    def __init__(self, horizon, K):
        super().__init__()
        self.horizon = horizon
        self.units = K
        self.gru_cell = GRUCell(K)
        self.gru_rnn = RNN(self.gru_cell, return_state=True)
        self.dense = Dense(1, activation='linear')

        self.compile(optimizer='adam', loss='mse')

    def warmup(self, inputs):

        y0_orig = inputs[1]

        y0 = tf.expand_dims(y0_orig, axis=1)

        x = inputs[0]

        x = x[:, 0:1, :]

        x = tf.concat([x, y0], axis=2)

        # inputs.shape => (batch, time, features)
        # x.shape => (batch, gru_units)
        x, *state = self.gru_rnn(x)

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

        y_pred, prediction, state = self.warmup(inputs)

        inputs = inputs[0]

        predictions = [prediction]

        for n in range(1, self.horizon):

            x = Lambda(lambda z: z[:, n, :])(inputs)

            x = Concatenate()([x, y_pred])

            x, state = self.gru_cell(x, states=state, training=training)

            prediction = self.dense(x)

            predictions.append(prediction)

            y_pred = tf.roll(y_pred, shift=1, axis=1)

            y_pred = Lambda(lambda z: z[:, 1:])(y_pred)

            y_pred = Concatenate()([prediction, y_pred])

        predictions = tf.stack(predictions)

        predictions = tf.transpose(predictions, [1, 0, 2])

        return predictions


class SimpleRNN(tf.keras.Model):
    """Implementação em classe da RNN standard utilizando a Functional API

    Para treinamento na configuração parallel (horizonte > 1)

    Não utilizado e desatualizado.

    Args:
            horizon (int): tamanho da janela de predição
            K (int): número de nodos na hidden layer
    """
    def __init__(self, horizon, K):
        super().__init__()
        self.horizon = horizon
        self.units = K
        self.simple_rnn_cell = SimpleRNNCell(K)
        self.simple_rnn_layer = RNN(
            self.simple_rnn_cell, return_state=True)
        self.dense = Dense(1, activation='linear')

        self.compile(optimizer='adam', loss='mse')

    def warmup(self, inputs):

        y0_orig = inputs[1]

        y0 = tf.expand_dims(y0_orig, axis=1)

        x = inputs[0]

        x = x[:, 0:1, :]

        x = tf.concat([x, y0], axis=2)

        # inputs.shape => (batch, time, features)
        # x.shape => (batch, gru_units)
        x, *state = self.simple_rnn_layer(x)

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

        y_pred, prediction, state = self.warmup(inputs)

        inputs = inputs[0]

        predictions = [prediction]

        for n in range(1, self.horizon):

            x = Lambda(lambda z: z[:, n, :])(inputs)

            x = Concatenate()([x, y_pred])

            x, state = self.simple_rnn_cell(x, states=state, training=training)

            prediction = self.dense(x)

            predictions.append(prediction)

            y_pred = tf.roll(y_pred, shift=1, axis=1)

            y_pred = Lambda(lambda z: z[:, 1:])(y_pred)

            y_pred = Concatenate()([prediction, y_pred])

        predictions = tf.stack(predictions)

        predictions = tf.transpose(predictions, [1, 0, 2])

        return predictions


def input_selection_incremental(data, output, params):
    """EM DESUSO! Essa é a versão incremental, originalmente utilizada, a
    versão decremental teve melhores resultados.
    Determina empiricamente o melhor conjunto de variáveis a ser utilizado
    como entrada no modelo da saída atual. Esta versão é incremental, ou seja,
    começa considerando ordem zero para todas variáveis de entrada (menos para
    a saída, para manter recursividade), e selecionando a com melhor
    desempenho em cada estágio.

    Args:
        data (pd.DataFrame): conjunto de dados retornado pela trim_data
        output (str): saída atual
        params (dict): diversos parâmetros de execução

    Returns:
        best_regressors (pd.Series): contém as input variables e suas ordens
        search_results (dict): dicionário contendo informações do treinamento,
        como tempo de execução e as escolhas feitas em cada estágio.
    """
    # inicia o relógio para marcar o tempo de execução
    start = time.time()

    # extrai os parâmetros do dicionário de parâmetros
    max_stages = params["max stages"]
    trains_per_option = params["trains per option"]
    search_patience = params["search patience"]
    max_epochs = params["max epochs"]
    patience = params["early stop patience"]
    partition_size = params["partition size"]
    K = params["hidden layer nodes"]
    horizon = params["horizon"]
    starting_y_order = params["starting order"]
    target_loss = params["target loss"]
    acceptable_loss = params["acceptable loss"]
    min_delta_loss = params["min delta loss"]
    validation_size = params["validation size"]
    structure = params["structure"]
    optimizer = params["optimizer"]
    loss = params["loss"]
    scaler = params["scaler"]

    if partition_size < 1:
        data, _ = np.split(data, [int(partition_size * len(data))])

    # grava os parâmetros que foram utilizados
    search_results = {"parameters": params}

    # o early stop é um callback do tf que para o treino assim que é detectado
    # overfitting
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                  patience=patience,
                                                  verbose=1)

    stages_without_improvement = 0

    variables = data.columns

    regressors = pd.Series(
        data=np.zeros(
            len(variables)),
        index=variables,
        dtype=int,
        name="ordem")

    if starting_y_order == 0:
        print("Starting y order must be > 0 for recursivity. Using order 1")
        starting_y_order = 1

    regressors[output] = starting_y_order

    # inicia os placeholders em valores altos para serem substituídos no
    # primeiro estágio
    best_loss = 999
    best_option_loss = 999

    for stage in range(max_stages):

        # cada variável representa uma opção, no caso a opção de incrementar a
        # respectiva ordem em 1
        for option in variables:

            testing_regressors = regressors.copy(deep=True)

            testing_regressors[option] += 1

            print("Starting stage " + str(stage + 1) +
                  ", option = " + option + ", testing:")
            print(testing_regressors.to_string())

            X, Y = du.build_sets(
                data, testing_regressors, output, scaler=scaler)

            if horizon > 1:
                X, Y, y0 = du.recursive_sets(
                    X, Y, output, horizon, testing_regressors[output],
                    shuffle=True)
            else:
                X, Y = du.shuffle_sets(np.array(X), np.array(Y))
                X, Y = np.array(X), np.array(Y)

            loss_sum_for_option = 0

            # faz varios treinos e tira a média, para ter resultados mais
            # consistentes
            for train in range(trains_per_option):

                if horizon > 1:

                    if structure == "DLP":
                        model = cm.ParallelModel(
                            horizon, K, loss=loss, optimizer=optimizer)
                    elif structure == "LSTM":
                        model = cm.LSTM(
                            horizon, K, loss=loss, optimizer=optimizer)

                    history = model.fit([X,
                                         y0],
                                        Y,
                                        epochs=max_epochs,
                                        validation_split=validation_size,
                                        verbose=0,
                                        callbacks=[early_stop])

                else:
                    model = cm.serial_parallel_model(
                        K, loss=loss, optimizer=optimizer)
                    history = model.fit(
                        X,
                        Y,
                        epochs=max_epochs,
                        validation_split=validation_size,
                        shuffle=True,
                        verbose=0,
                        callbacks=[early_stop])

                if validation_size > 0:
                    loss_for_training = history.history['val_loss'][-1]
                else:
                    loss_for_training = history.history['loss'][-1]

                print("Loss for training = " + str(loss_for_training))

                loss_sum_for_option += loss_for_training

                del model

            # Calcula a média do loss para a opção
            loss_for_option = loss_sum_for_option / trains_per_option

            print("Mean loss for option = " + str(loss_for_option))

            if (option == 'u1') or (loss_for_option < best_option_loss):

                best_option = option
                best_option_loss = loss_for_option
                best_option_regressors = testing_regressors.copy(deep=True)

        # Ao sair do loop, pega o melhor regressors dentre as opções
        regressors = best_option_regressors.copy(deep=True)

        print("Best option: " +
              best_option +
              ", 'regressors' = " +
              str(regressors.to_numpy()))

        search_results["Stage " + str(stage + 1)] = "Best option: " + str(
            regressors.to_numpy()) + ", average loss: " + str(best_option_loss)

        # Verifica se o resultado da melhor opção é o melhor overall
        if (best_option_loss + min_delta_loss < best_loss):

            stages_without_improvement = 0
            best_regressors = regressors.copy(deep=True)
            best_loss = best_option_loss
            print("Best regressors updated")

        elif (best_loss < acceptable_loss):
            print("Best option doesn't improve performance and acceptable \
                  loss was already reached, interrupting search")
            stop_reason = "No significant improvement while acceptable loss"
            break
        else:
            print(
                "Best option doesn't improve performance and acceptable loss \
                was not reached, continuing search")
            stages_without_improvement += 1

        if (best_loss < target_loss):
            print(
                "Target loss " +
                str(target_loss) +
                " reached, interrupting search")
            stop_reason = "target loss reached"
            break
        elif (stages_without_improvement == search_patience + 1):
            print("Interrupting search for not achieving improvements in " +
                  str(search_patience + 1) + " stages")
            stop_reason = "stage patience run out"
            break
        elif (stage == max_stages - 1):
            print("Last stage completed")
            stop_reason = "Last stage completed"

    print("Best regressors found:")
    print(best_regressors.to_string())

    selected_regressors = []

    # Dropa as variáveis com ordem 0 e preenche a lista selected_regressors
    # para incluir no dicionário
    for variable in variables:

        if best_regressors[variable] == 0:
            best_regressors.drop(variable, inplace=True)
        else:
            selected_regressors.append(
                variable + " = " + str(best_regressors[variable]))

    search_results["selected regressors"] = selected_regressors

    search_results["stop reason"] = stop_reason

    search_results["execution time"] = str(
        (time.time() - start) / 60) + " minutes"

    print("Execution time: " + search_results["execution time"])

    return best_regressors, search_results
