# -*- coding: utf-8 -*-
"""
Programa criado para inspeções/testes separados.
Permite ver a correlação entre as variáveis de entrada (determinadas
por um 'regressors') e a variável de saída do sistema.
É utilizado para filtrar variáveis inúteis no início do input selection,
acelerando o processo de criação dos modelos.
Lembrar que (alta) correlação não implica em causalidade, mas baixa correlação,
nesta atividade, deve indicar que a variável não influencia na saída.
"""
import tensorflow as tf
import numpy as np
import pandas as pd
from collections import Counter
# Para conseguir importar o data_utils estando no folder misc
import sys
sys.path.insert(0, '..')
from utils import data_utils as du


tf.keras.backend.clear_session()
tf.random.set_seed(0)

min_abs_corr = 0.3
output = 'y1'
raw_data, variable_names, _ = du.load_data("emso", root="..")

# é possível comentar o trim_data, para considerar todas as variáveis
data = raw_data.copy(deep=True)
# data = du.trim_data(raw_data, output)

# o número que multiplica np.ones pode ser alterado, ele define até que
# ordem serão consideradas as entradas. Normalmente a primeira ordem
# é a mais relevante, as demais são bem parecidas
regressors = pd.Series(data=3*np.ones(len(data.columns)),
                       index=data.columns, dtype=int, name="order")

X, Y = du.build_sets(data, regressors, output,
                     return_Y_scaler=False, scaler="Standard")

# .corr() calcula a correlação entre as variáveis
# [output + '(k)'] pega somente a coluna da saída do sistema
# dropna() remove valores nan (ocorrem qdo acontece divisão por 0)
correlation = pd.concat([X, Y], axis=1).corr()[output + '(k)'].dropna()

# dropa o a correlação da saída(k) com ela mesma, que é sempre 1
correlation.drop(correlation.tail(1).index, inplace=True)

filtered = correlation.abs().where(lambda x: x >= min_abs_corr).dropna()

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
