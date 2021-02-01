"""Módulo de funções para criação e treinamento de modelos

Contém os algoritmos de input selection e K selection, para seleção da
estrutura dos modelos de rede neural.
"""

from . import data_utils as du
from .custom_models import ParallelModel, serial_parallel_model, LSTM
import numpy as np
import time
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split


def create_models(exec_cfg, training_dictionary, raw_data):

    for y in range(exec_cfg["first y"], exec_cfg["last y"] + 1):

        output = 'y' + str(y)

        print("Initializing training operations for output " + output)

        try:
            dependency = training_dictionary[output]["dependency mask"]
        except KeyError:
            dependency = None
            try:
                training_dictionary[output]
            except KeyError:
                training_dictionary[output] = {}
                print(
                    "No dictionary entry for " +
                    output +
                    ", creating a new entry")

        if exec_cfg["input selection params"]["use all variables"]:
            data = raw_data
        else:
            data = du.trim_data(raw_data, output, dependency)

        if exec_cfg["find inputs"]:
            regressors, input_selection_results = \
                input_selection_decremental(
                    data, output, exec_cfg["input selection params"])

            training_dictionary[output]["input selection results"] = \
                input_selection_results
            training_dictionary[output]["model"] = {"regressors": regressors}

            du.save_pickle(training_dictionary, "analysis\\dictionary.pickle")
        else:
            try:
                regressors = training_dictionary[output]["model"]["regressors"]
            except KeyError:
                print("No regressors found for " + output +
                      ", please run input selection first")

        if exec_cfg["find K"]:
            K, weights, K_selection_results = K_selection(
                data, regressors, output, exec_cfg["K selection params"])

            training_dictionary[output]["K selection results"] = \
                K_selection_results
            training_dictionary[output]["model"]["K"] = K
            training_dictionary[output]["model"]["weights"] = weights

            du.save_pickle(training_dictionary, "analysis\\dictionary.pickle")

    return training_dictionary


def input_selection_decremental(data, output, params):
    """Determina empiricamente o melhor conjunto de variáveis a ser utilizado
    como entrada no modelo da saída atual. Esta versão é decremental, ou seja,
    começa considerando starting_order para todas variáveis de entrada, e
    decrementa uma das variáveis por estágio, escolhendo a que mais atrapalha
    o desempenho, ficando somente as variáveis importantes no final, com a
    ordem correta.

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
    max_epochs = params["max epochs"]
    search_patience = params["search patience"]
    patience = params["early stop patience"]
    partition_size = params["partition size"]
    K = params["hidden layer nodes"]
    horizon = params["horizon"]
    # nessa versão todos inputs começam com essa ordem
    starting_order = params["starting order"]
    target_loss = params["target loss"]
    acceptable_loss = params["acceptable loss"]
    min_delta_loss = params["min delta loss"]
    validation_size = params["validation size"]
    structure = params["structure"]
    optimizer = params["optimizer"]
    loss = params["loss"]
    scaler = params["scaler"]

    # Para casos em que os dados são muito grandes (100k+), pode-se pegar
    # apenas uma parte
    if partition_size < 1:
        data, _ = np.split(data, [int(partition_size * len(data))])

    # grava os parâmetros que foram utilizados
    search_results = {"parameters": params}

    # o early stop é um callback do tf que para o treino assim que é detectado
    # overfitting
    early_stop = EarlyStopping(monitor='loss', patience=patience, verbose=1)

    variables = data.columns

    regressors = pd.Series(
        data=np.ones(len(variables)) * starting_order,
        index=variables, dtype=int, name="ordem")

    # Calcula performance com regressors inicial
    print("Testing initial regressors")
    print(regressors.to_string())
    X, Y = du.build_sets(data, regressors, output, scaler=scaler)
    if horizon > 1:
        X, Y, y0 = du.recursive_sets(
            X, Y, output, horizon, regressors[output], shuffle=True)
    else:
        X, Y = du.shuffle_sets(np.array(X), np.array(Y), return_array=True)

    loss_sum_for_initial = 0

    for train in range(trains_per_option):

        # Se horizon == 1, usa o modelo da API Sequential, que executa bem mais
        # rápido
        if horizon > 1:
            if structure == "DLP":
                model = ParallelModel(
                    horizon, K, loss=loss, optimizer=optimizer)
            elif structure == "LSTM":
                model = LSTM(
                    horizon, K, loss=loss, optimizer=optimizer)
            history = model.fit([X, y0],
                                Y,
                                epochs=max_epochs,
                                validation_split=validation_size,
                                verbose=0,
                                callbacks=[early_stop])
        else:
            # É bem mais rápido usar o modelo definido com a Sequential API
            # quando o horizonte = 1, por isso esses if-elses
            model = serial_parallel_model(
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

        loss_sum_for_initial += loss_for_training

        del model

    loss_for_initial = loss_sum_for_initial / trains_per_option
    print("Mean loss for initial regressors = " + str(loss_for_initial))

    # inicia os melhores valores como os obtidos com o regressors inicial
    best_loss = loss_for_initial
    best_regressors = regressors.copy(deep=True)
    # serve para escolher a melhor opção ao final de um estágio
    best_option_loss = None
    stages_without_improvement = 0

    for stage in range(max_stages):

        # Flag importante para sinalizar um novo estágio
        new_stage = True

        # cada variável representa uma opção, no caso a opção de decrementar a
        # sua ordem em 1, a opção que tiver best_option_loss é escolhida
        for option in variables:

            testing_regressors = regressors.copy(deep=True)

            # se a variável respectiva à opção já tem ordem 0 (ou ordem 1 se
            # for a saída), vai para a próxima opção
            if testing_regressors[option] == 0 or (
                    option == output and testing_regressors[output] == 1):
                continue  # 'pula' o loop

            testing_regressors[option] -= 1

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
                X, Y = du.shuffle_sets(np.array(X), np.array(Y),
                                       return_array=True)

            loss_sum_for_option = 0

            # faz vários treinos e tira a média, para ter resultados mais
            # consistentes, devido as inicializações serem randômicas
            for train in range(trains_per_option):

                if horizon > 1:

                    if structure == "DLP":
                        model = ParallelModel(
                            horizon, K, loss=loss, optimizer=optimizer)
                    elif structure == "LSTM":
                        model = LSTM(
                            horizon, K, loss=loss, optimizer=optimizer)

                    history = model.fit([X,
                                         y0],
                                        Y,
                                        epochs=max_epochs,
                                        validation_split=validation_size,
                                        verbose=0,
                                        callbacks=[early_stop])

                # usa a modelo Sequential API quando horizon = 1, cuja
                # execução é consideravelmente mais rápida
                else:
                    model = serial_parallel_model(
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

            if new_stage or loss_for_option < best_option_loss:
                new_stage = False
                best_option_loss = loss_for_option
                best_option_regressors = testing_regressors.copy(deep=True)

            if best_option_loss + min_delta_loss < best_loss:

                best_regressors = best_option_regressors.copy(deep=True)
                best_loss = loss_for_option
                print("Best regressors updated")
                break

        regressors = best_option_regressors.copy(deep=True)

        print("STAGE ====== " + str(stage))

        # se a loss da melhor opção recebeu a melhor loss overall, significa
        # que não houve melhoria significativa de desempenho nesse estágio
        if best_option_loss != best_loss:
            stages_without_improvement += 1
            print("No significant improvements in this stage")
        else:
            stages_without_improvement = 0

        if stages_without_improvement == search_patience + 1:
            print(f"{search_patience + 1} stages without improvement, " +
                  "interrupting search")
            stop_reason = "search patience run out"
            break

        if (best_loss <= target_loss):
            print(f"Target loss {target_loss} reached, interrupting search")
            stop_reason = "target loss reached"
            break

        elif (best_loss <= acceptable_loss and
              stages_without_improvement == 1):
            print("No option improves performance and acceptable loss has \
                  already been reached, interrupting search")
            stop_reason = "acceptable loss"
            break

        elif (stage == max_stages - 1):
            print("Last stage completed")
            stop_reason = "Last stage completed"
            break

        elif (sum(regressors) == 1):
            print("Tested all possible regressors from the initial regressors")
            stop_reason = "tested all regressors"
            break

        # print("Best option: " + best_option + ", 'regressors' = " +
        #       str(regressors.to_numpy()) )
        # search_results["Stage " + str(stage+1)] = "Best option: " +
        # str(regressors.to_numpy()) + ", average loss: " +
        # str(best_option_loss)

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


def K_selection(data, regressors, output, params):
    """Seleciona o número de nodos (K) na hidden layer do DLP. Começa criando
    uma rede com K_min, treina trains_per_K e guarda o melhor desempenho,
    incrementa K e repete o processo, até atingir uma condição de parada.

    Args:
        data (pd.DataFrame): conjunto de dados
        regressors (pd.Series): input variables escolhidas no input selection
        output (str): saída atual
        params (dict): parâmetros de execução

    Returns:
        best_K (int): melhor valor determinado para K
        best_weights (list): parâmetros treinados da rede com best_K nodes
        search_results (dict): dicionário contendo informações do treinamento
    """
    start = time.time()

    K_min = params["K min"]
    K_max = params["K max"]
    trains_per_K = params["trains per K"]
    search_patience = params["search patience"]
    max_epochs = params["max epochs"]
    partition_size = params["partition size"]
    patience = params["early stop patience"]
    horizon = params["horizon"]
    target_loss = params["target loss"]
    min_delta_loss = params["min delta loss"]
    validation_size = params["validation size"]
    test_size = params["test size"]

    if partition_size < 1:
        data, _ = np.split(data, [int(partition_size * len(data))])

    # grava os parâmetros que foram utilizados
    search_results = {"parameters": params}

    # o early stop é um callback do tf que para o treino assim que é detectado
    # overfitting
    early_stop = EarlyStopping(monitor='loss', patience=patience, verbose=1)

    best_loss = None
    best_loss_for_K = None
    stages_without_improvement = 0
    stop_reason = "K max reached"  # default, para registro

    X, Y = du.build_sets(data, regressors, output)

    if horizon > 1:
        X, Y, y0 = du.recursive_sets(
            X, Y, output, horizon, regressors[output], shuffle=True)
    else:
        X, Y = du.shuffle_sets(np.array(X), np.array(Y))
        X, Y = np.array(X), np.array(Y)

    """
    Essa chamada faz o split do test set, para ser utilizada ao final para
    avaliação do melhor modelo obtido. O split de treino/validação é feito
    pelo método model.fit quando os modelos são treinados. Como shuffle já
    foi feito na recursive_sets, aqui é necessário somente particionamento.
    """
    if horizon > 1 and test_size is not False:
        X, X_test, Y, Y_test, y0, y0_test = train_test_split(
            X, Y, y0, test_size=test_size, random_state=42)
    elif test_size is not False:
        X, X_test, Y, Y_test = train_test_split(
            X, Y, test_size=test_size, random_state=42)

    # a busca começa em K_min e vai até K_max, onde K é o número de neurônios
    # na camada oculta
    for K in range(K_min, K_max + 1):

        print("Testing model with " + str(K) + " hidden nodes")

        # para ter resultados mais consistentes, cada K é inicializado algumas
        # vezes, só a melhor inicialização é considerada
        for initialization in range(trains_per_K):

            if horizon > 1:
                model = ParallelModel(horizon, K)
                history = model.fit([X,
                                     y0],
                                    Y,
                                    validation_split=validation_size,
                                    epochs=max_epochs,
                                    verbose=0,
                                    callbacks=[early_stop])
            else:
                model = serial_parallel_model(K)
                history = model.fit(
                    X,
                    Y,
                    validation_split=validation_size,
                    shuffle=True,
                    epochs=max_epochs,
                    verbose=0,
                    callbacks=[early_stop])

            loss_for_init = history.history['val_loss'][-1]

            # loss_for_init = history.history['loss'][-1]

            if (initialization == 0) or (loss_for_init < best_loss_for_K):
                best_loss_for_K = loss_for_init

                best_weights_for_K = model.get_weights()

            del model

        search_results["K = " + str(K)] = "Best loss: " + str(best_loss_for_K)

        if (K == K_min) or (best_loss_for_K + min_delta_loss < best_loss):

            stages_without_improvement = 0

            best_loss = best_loss_for_K

            best_K = K

            best_weights = best_weights_for_K.copy()

        else:
            stages_without_improvement += 1
            print("No significant performance improvement with K = " + str(K))

        if (stages_without_improvement == search_patience + 1):
            print("Stopping at K = " +
                  str(K) +
                  " for not achieving improvements in " +
                  str(search_patience +
                      1) +
                  " stages")
            stop_reason = "K patience run out"
            break
        elif (best_loss < target_loss):
            print(
                "Stopping at K = " +
                str(K) +
                " for reaching target loss value")
            stop_reason = "target loss reached"
            break

    if test_size is not False:
        print("Testando modelo obtido no conjunto de teste")
        model = ParallelModel(horizon, best_K)
        # Força o custom model a definir a input layer shape, possibilitando
        # setar os weights
        model.predict([X[0:1], y0[0:1]], verbose=False)
        model.set_weights(best_weights)
        search_results["Loss in test set"] = model.evaluate(
            [X_test, y0_test], Y_test)

    search_results["best K"] = best_K

    search_results["stop reason"] = stop_reason

    search_results["execution time"] = str(
        (time.time() - start) / 60) + " minutes"

    print("Execution time: " + search_results["execution time"])

    return best_K, best_weights, search_results
