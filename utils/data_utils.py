# -*- coding: utf-8 -*-
"""Módulo de carregamento e preparação de dados, também tem funções para
o carregamento e salvamento de objetos .pickle
"""

import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import sklearn.utils
import scipy.io
import pandas as pd
import math
from collections import Counter

# Função utilizada para carregar os dados de treinamento


def load_data(batch="batch_1", root="."):
    """Função utilizada para carregar os dados de treinamento

    Opções atuais: "batch_1", "batch_2", "batch_3", "emso" ou "simulink".
    É necessário criar uma nova opção para importar novos dados. Deve retornar
    um DataFrame com as colunas no formato "u1, u2, ..., uNu, y1, y2, ...,
    yNy".

    Args:
        batch (str, optional): Defaults to "batch_1".
        root (str, optional): Root directory. Defaults to ".".

    Returns:
        raw_data (pd.DataFrame): contém as amostras das variáveis do sistema
        variable_names (dict): dicionário contendo o código de identificação
        de cada variável, exemplo: u1 -> XIA2017A
        Ny (int): Número de saídas do sistema
    """
    # Esse é o primeiro lote que a Carolina me enviou
    # Contém dados reais do processo para efetuar o treinamento das redes
    # Contém 99.1k amostras
    # Tem 5 variáveis 'u' e 12 variáveis 'y'
    if batch == "batch_1":

        input_ = scipy.io.loadmat(root + "\\data\\batch_1\\input.mat")
        output = scipy.io.loadmat(root + "\\data\\batch_1\\output.mat")

        raw_data = []
        variable_names = {}

        u, y = 1, 1

        for key, value in input_.items():
            if isinstance(value, np.ndarray):
                raw_data.append(value.flatten())
                variable_names["u" + str(u)] = key
                u += 1

        for key, value in output.items():
            if isinstance(value, np.ndarray):
                raw_data.append(value.flatten())
                variable_names["y" + str(y)] = key
                y += 1

        Ny = 12

        raw_data = np.transpose(np.array(raw_data))

        raw_data = pd.DataFrame(raw_data, columns=list(variable_names.keys()))

    # Segundo lote com dados reais do processo
    # Contém 205.3k de amostras, porém alguns zeros abrutos nas variáveis
    # Tem 5 variáveis 'u' e 12 variáveis 'y'
    elif batch == "batch_2":

        inputs = pd.read_hdf(root + "\\data\\batch_2\\input_validation.h5")

        outputs = pd.read_hdf(root + "\\data\\batch_2\\output_validation.h5")

        raw_data = pd.concat([inputs, outputs], axis=1)

        raw_data.reset_index(drop=True, inplace=True)

        variable_names = [list(raw_data.columns), []]

        for i in range(1, inputs.shape[1] + 1):
            variable_names[1].append("u" + str(i))

        for i in range(1, outputs.shape[1] + 1):
            variable_names[1].append("y" + str(i))

        raw_data.columns = variable_names[1]

        Ny = 12

    # Terceiro lote com dados reais do processo
    # Tem 7 variáveis 'u' e 4 variáveis 'y'
    # Requer uma etapa de pre-processamento para tirar os valores nan e
    # sincronizar as amostras. Contém 99.1k amostras
    elif batch == "batch_3":

        inputs = pd.read_hdf(root + "\\data\\batch_3\\inputs.h5").dropna()

        outputs = pd.read_hdf(root + "\\data\\batch_3\\outputs.h5")

        outputs.drop(outputs.head(2).index, inplace=True)

        raw_data = pd.concat([inputs, outputs], axis=1)

        raw_data.reset_index(drop=True, inplace=True)

        variable_names = [list(raw_data.columns)]

        variable_names.append(
            ['u1', 'u2', 'u3', 'u4', 'u5', 'u6', 'u7', 'y1', 'y2', 'y3', 'y4'])

        raw_data.columns = variable_names[1]

        Ny = 4

    # Dados do sistema de testes feito no Simulink com o intuito de validar os
    # algoritmos, em especial o de input selection. Contém 10k amostras
    # Regressors utilizado para gerar os dados = [u1 = 1, u2 = 0, u3 = 0, u4 =
    # 2, y1 = 3]
    elif batch == "simulink":

        raw_data = scipy.io.loadmat(
            root + '\\data\\simulink\\dados_simulink.mat')['simout']

        raw_data = pd.DataFrame(
            data=raw_data, columns=[
                'u1', 'u2', 'u3', 'u4', 'y1'])

        variable_names, Ny = None, 1

    # Dados do simluador, legados da iteração anterior do projeto
    # Foram levantados no software EMSO
    # Contém 18k amostras
    elif batch == "emso":

        raw_data = np.transpose(
            np.array(
                load_pickle(root +
                            "\\data\\emso_simulator\\dados18k_p74.pickle")))

        columns = load_pickle(root + "\\data\\emso_simulator\\columns.pickle")

        raw_data = pd.DataFrame(data=raw_data, columns=columns)

        variable_names = load_pickle(
            root + "\\data\\emso_simulator\\variable_names.pickle")

        Ny = 36

    return raw_data, variable_names, Ny


def load_pickle(path):
    """Função utilizada para carregar arquivos .pickle

    Args:
        path (str): caminho até o arquivo .pickle

    Returns:
        stuff (object): objeto Python a ser retornado
    """
    try:
        f = open(path, "rb")
    except IOError:
        stuff = {}
    else:
        stuff = pickle.load(f)
        f.close()
    return stuff


def save_pickle(stuff, path):
    """Função utilizada para salvar arquivos .pickle

    Args:
        stuff (object): objeto Python a ser salvo
        path (str): caminho até o arquivo .pickle a ser criado/sobreescrito
    """
    pickle.dump(stuff, open(path, "wb"))


def trim_data(raw_data, output, dependency=None):
    """Retorna o conjunto de dados apenas com as variáveis que serão
    utilizadas na rede neural da saída atual. Se não se sabe quais são estas
    variáveis, 'dependency' não é informada e assume-se que a saída é
    influenciada apenas pelas variáveis manipuladas 'u' e por ela mesma.

    Args:
        raw_data (pd.DataFrame): conjunto de dados
        output (str): saída atual
        dependency (list, optional): variáveis que influenciam na saída atual.
        Defaults to None.

    Returns:
        pd.DataFrame: raw_data só com as colunas importantes
    """
    if dependency is None:

        variaveis_influentes = [col for col in raw_data if col.startswith('u')]

        variaveis_influentes.append(output)

        data = raw_data[variaveis_influentes]

    else:
        data = raw_data[dependency]

    return data


def filter_by_correlation(min_abs_corr, X, Y, output):
    """Calcula a correlação entre as variáveis de entrada (determinadas
    por um 'regressors') e a variável de saída do sistema (output), então
    filtrar (elimina) as variáveis cuja correlação for menor do que
    min_abs_corr.

    Esta função é chamada no início do input selection, e
    acelerando o processo de criação dos modelos.

    Lembrar que (alta) correlação não implica em causalidade, mas uma baixa
    correlação indica que a variável não tem influência na saída.

    Args:
        min_abs_corr (float): menor correlação aceitável, valor absoluto em
        porcentagem
        X (pd.DataFrame): conjunto de entradas
        Y (pd.DataFrame): conjunto de saídas esperadas
        output (string): saída para a qual o modelo está sendo criado

    Returns:
        regressors (pd.Series): novo regressors filtrado
        no_filtered (int): qtd de variáveis filtradas
    """

    correlation = pd.concat([X, Y], axis=1).corr()[output + '(k)'].dropna()

    # dropa o a correlação da saída(k) com ela mesma, que é sempre 1
    correlation.drop(correlation.tail(1).index, inplace=True)

    filtered = correlation.abs().where(lambda x: x >= min_abs_corr).dropna()

    # número de variáveis filtradas, apenas para printar
    no_filtered = len(correlation) - len(filtered)

    # constrói uma lista só com os índices de "filtered", sem os "(k-...)"
    indexes = []

    for index in list(filtered.index):
        indexes.append(index.split('(')[0])

    # infere as ordens ao contar o número de duplicatas, criando um dicionário
    # que já é o novo regressors, com {key : value} --> {variável : ordem}
    regressors = dict(Counter(indexes))

    # só falta transformá-lo em um pandas Series e retornar
    regressors = pd.Series(data=list(regressors.values()),
                           index=list(regressors.keys()),
                           dtype=int, name="order")

    return regressors, no_filtered


def build_sets(
        data,
        regressors,
        output,
        return_Y_scaler=False,
        scaler="MinMax"):
    """A função build_sets extrai o máximo de exemplos possíveis de raw_data,
    levando em conta as ordens das variáveis de entrada da rede, informada no
    'regressors'.

    Exemplo: se output = 'y1', e regressors = [u1 = 1, u2 = 2, y1 = 3],

    y1(k) = f(u1(k-1), u2(k-1), u2(k-2), y1(k-1), y1(k-2), y1(k-3))

    Args:
        data (pd.DataFrame): conjunto de dados
        regressors (pd.Series): variáveis de entrada e respectivas ordens
        output (str): saída atual
        return_Y_scaler (bool, optional): retorna scaler, só usado para plots.
        scaler (str, optional): scaler, pode ser MinMax ou Standard.

    Returns:
        X (pd.DataFrame): DataFrame com os exemplos extraídos de raw_data
        Y (pd.DataFrame): DataFrame de uma coluna com as labels Y
        Y_scaler (optional): scaler do scikit-learn
    """

    maior_ordem = max(regressors)

    X = pd.DataFrame()

    Y = data[output].copy().shift(-maior_ordem).dropna() \
        .reset_index(drop=True).rename(output + "(k)").to_frame()

    for variavel, ordem in regressors.items():

        for atraso in range(1, ordem + 1):

            dados_variavel = data[variavel].copy().shift(
                atraso - maior_ordem).dropna().reset_index(drop=True)

            X[variavel +
              "(k-" +
              str(atraso) +
                ")"] = dados_variavel.drop(dados_variavel.tail(1).index)

    # Scales and translates each feature individually between zero and one.
    if scaler == "MinMax":
        X[X.columns] = MinMaxScaler().fit_transform(X[X.columns])
        Y_scaler = MinMaxScaler().fit(Y)
        Y[Y.columns] = Y_scaler.transform(Y[Y.columns])

    # Standardize features by removing the mean and scaling to unit variance
    elif scaler == "Standard":
        X[X.columns] = StandardScaler().fit_transform(X[X.columns])
        Y_scaler = StandardScaler().fit(Y)
        Y[Y.columns] = Y_scaler.transform(Y[Y.columns])

    if return_Y_scaler:
        return X, Y, Y_scaler
    else:
        return X, Y


def recursive_sets(X, Y, output, horizon, y_order, shuffle=False):
    """Essa função prepara os dados para treinamento na configração Parallel,
    cada exemplo não é mais uma única amostra, mas uma sequência de 'horizon'
    amostras. Para isso, X e Y são transformados em conjuntos tridimensionais.
    E.g., se existem 10k dados e o horizonte de predição escolhido é 50,
    teremos 200 exemplos de 50 janelas (amostras em sequência).

    Args:
        X (pd.DataFrame): valores de entrada para cada exemplo
        Y (pd.DataFrame): respectivo valor esperado na saída
        output (str): saída atual
        horizon (int): horizonte de predição/número de timesteps
        y_order (int): ordem da variável de saída
        shuffle (bool, optional): embaralha as janelas, deve melhorar o
        desempenho. Defaults to False.

    Returns:
        X (np.array): array no formato (exemplo, timestep, variável)
        Y (np.array): array no formato (exemplo, timestep, 1)
        y0 (np.array): vetor que contém os estados iniciais dos regressores
        da saída, para cada janela.
    """

    # len(x) seria igual ao número de janelas para um horizon = 1
    num_janelas = math.floor(len(X) / horizon)

    # Se a divisão não for exata, são removidas as últimas
    # amostras até a conta fechar. Não tem problema em fazer isso, pois
    # geralmente são muitas amostras.
    # Valores a remover para que cada janela tenha o mesmo número de exemplos
    # Essa é uma estratégica melhor do que repetir o último valor
    remocoes = int((len(X) / horizon) * horizon - num_janelas * horizon)

    X.drop(X.tail(remocoes).index, inplace=True)
    Y.drop(X.tail(remocoes).index, inplace=True)

    # Para X e Y, coloca as amostras numa lista com 'num_janelas' exemplos de
    # sequências de 'horizon' amostragens
    X = np.split(X, num_janelas)
    Y = np.split(Y, num_janelas)

    if shuffle:
        X, Y = shuffle_sets(X, Y)

    # y0 contém os valores de y(k-1)...(k-y_order)
    # Existe uma linha em y0 para cada num_janelas
    # 'horizon' não participa das contas, pois os valores de y0 só são
    # utilizados no primeiro instante, servindo como estado inicial
    y0 = np.zeros((num_janelas, y_order))

    # Preenche y0 a partir dos regressores de y1 contidos em X
    for i in range(num_janelas):
        for j in range(y_order):
            y0[i, j] = X[i].head(1)[output + "(k-" + str(j + 1) + ')']

        # Agora os regressores de y1 em X devem ser removidos, pois estes serão
        # informados por y0
    for janela in X:
        for i in range(y_order):
            janela.drop(output + "(k-" + str(i + 1) + ')',
                        inplace=True, axis=1)

    return np.array(X), np.array(Y), y0


def shuffle_sets(X, Y, return_array=False):
    """Embaralha os conjuntos X e Y. Deve melhorar a generalização
    dos modelos criados, pois após o split, os conjuntos de treino,
    teste e validação ficam mais homogêneos.

    Args:
        X (pd.DataFrame): conjunto de entradas
        Y (pd.DataFrame): conjunto de saídas esperadas
        return_array (bool, optional): Se true, casta um np.array() sobre
        X e Y antes de retornar. Usado pelo modelo serial-parallel (horionte
        1), pois ele não chama a recursive_sets(), mas usa a shuffle_sets()
        de forma externa. Defaults to False.

    Returns:
        X (np.array or pd.DataFrame): X embaralhado
        Y (np.array or pd.DataFrame): Y embaralhado
    """

    index_shuf = list(range(len(X)))

    index_shuf = sklearn.utils.shuffle(index_shuf, random_state=42)

    X = [X[i] for i in index_shuf]

    Y = [Y[i] for i in index_shuf]

    if return_array:
        return np.array(X), np.array(Y)
    else:
        return X, Y
