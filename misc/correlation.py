""" Permite ver a correlação entre as variáveis para uma determinada saída.

Poderia ser utilizado para filtrar algumas variáveis para o input selection,
ao invés de usar sempre o padrão (considerar todas as entradas e a própria
saída, ignorando as outras saídas).
Lembrar que correlação não implica em causalidade.
"""
import tensorflow as tf
import numpy as np
import pandas as pd
# Para conseguir importar o data_utils estando no folder misc
import sys
sys.path.insert(0, '..')
from utils import data_utils as du
tf.keras.backend.clear_session()
tf.random.set_seed(0)

output = 'y1'
raw_data, variable_names, _ = du.load_data("batch_1")
data = du.trim_data(raw_data, output)
regressors = pd.Series(data=5 * np.ones(len(data.columns)),
                       index=data.columns, dtype=int, name="order")

X, Y = du.build_sets(data, regressors, output,
                     return_Y_scaler=False, scaler="Standard")

correlation = pd.concat([X, Y], axis=1).corr()[output + '(k)'].dropna()
