# -*- coding: utf-8 -*-
"""
Este programa permite comparar três modelos criados e treinados com valores
diferentes de horizontes, comparando-os em um horizonte em comum.
Naturalmente, os modelos devem usar o mesmo conjunto de dados, saída e
entrada (regressors).
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
import scipy.io
import matplotlib.pyplot as plt
from matplotlib import use
import datetime
# Para conseguir importar data_utils estando no folder misc
import sys
sys.path.insert(0, '..')
from utils import data_utils as du
from utils import custom_models as cm


tf.keras.backend.clear_session()
tf.random.set_seed(0)
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)
tf.autograph.experimental.do_not_convert

output = "y1"
raw_data, variable_names, _ = du.load_data("batch_1", root="..")
data = du.trim_data(raw_data, output)
regressors = pd.Series(data=3*np.ones(len(data.columns)),
                       index=data.columns,
                       dtype=int,
                       name="order")

X_orig, Y_orig, Y_scaler = du.build_sets(data,
                                         regressors,
                                         output,
                                         return_Y_scaler=True,
                                         scaler="Standard")

# Da forma como está, é possível comparar 3 modelos, treinados com 3 valores
# diferentes para horizon. Um dos modelos comparados tem estrutura series-
# parallel (horizon = 1), portanto só o horizonte dos dois modelos parallel
# são definidos aqui
horizon_model_parallel_1 = 5
horizon_model_parallel_2 = 15

# Esse modelo de plot define o horizonte em comum no qual serão comparados
# os outros 3 modelos. É o horizonte utilizado no plot e também nos cálculos
# dos valores de sum of squared error.
horizon_plot = 10

# Define otimizador e função de custo utilizados nos três modelos
# testar no optimizer adam: epsilon 1.0 e 0.1, o default é 1e-07
# testar sgd: momentum 0.5, 0.8, o default é 0.
optimizer = tf.keras.optimizers.Adam()
loss = 'mse'
# loss = cm.SSE()

# Todos modelos tem as mesmas entradas (usam o mesmo regressors) e o mesmo
# número de nodos na hidden layer (K), é por isso que podem ser comparados
# no mesmo horizonte de predição (definido em horizon_plot).
K = 8

max_epochs = 1000
early_stop = EarlyStopping(monitor='loss', patience=3, verbose=1)
# early_stop = EarlyStopping(monitor='val_loss', patience=3, verbose=1,
#                            mode='min', restore_best_weights=True)


# log_dir = "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
#                                                       histogram_freq=1)

""" MODELO PARA OS PLOTS """

X_plot, Y_plot, y0_plot = du.recursive_sets(X_orig.copy(deep=True),
                                            Y_orig.copy(deep=True),
                                            output,
                                            horizon_plot,
                                            regressors[output])

plot_model = cm.ParallelModel(horizon_plot, K)

# inicializa o plot_model, permitindo que ele receba os pesos dos outros
# modelos com set_weights
plot_model.predict([X_plot[0:1], y0_plot[0:1]], verbose=False)


# função que recebe os pesos dos modelos treinados e plota no horizonte
# de predição comum para comparações
def predictOnPlotHorizon(weights):
    plot_model.set_weights(weights)
    return plot_model.predict([X_plot, y0_plot], verbose=False)


# aqui serão armazenadas as predições de cada um dos 3 modelos
# no horizonte em comum, para plotagem e cálculo dos SSEs
predictions = []

""" MODELO 1 (SERIAL-PARALLEL) (horizonte = 1) """
print("Treinando modelo 1")

model = cm.serial_parallel_model(K, optimizer=optimizer, loss=loss)

model.fit(np.array(X_orig.copy(deep=True)),
          np.array(Y_orig.copy(deep=True)),
          epochs=max_epochs,
          verbose=0,
          callbacks=[early_stop])

print("Efetuando predições com modelo 1")
predictions.append(predictOnPlotHorizon(model.get_weights()))

""" MODELO 2 (PARALLEL) """
print("Treinando modelo 2")

model = cm.ParallelModel(horizon=horizon_model_parallel_1,
                         K=K,
                         optimizer=optimizer,
                         loss=loss)

X, Y, y0 = du.recursive_sets(X_orig.copy(deep=True),
                             Y_orig.copy(deep=True),
                             output,
                             horizon_model_parallel_1,
                             regressors[output])

history = model.fit([X, y0],
                    Y,
                    epochs=max_epochs,
                    verbose=0,
                    callbacks=[early_stop])

print("Efetuando predições com modelo 2")
predictions.append(predictOnPlotHorizon(model.get_weights()))

# tensorboard --logdir logs/fit
# tf.keras.utils.plot_model(model)

""" MODELO 3 (PARALLEL) """
print("Treinando modelo 3")

model = cm.ParallelModel(horizon=horizon_model_parallel_2,
                         K=K,
                         optimizer=optimizer,
                         loss=loss)

X, Y, y0 = du.recursive_sets(X_orig.copy(deep=True),
                             Y_orig.copy(deep=True),
                             output,
                             horizon_model_parallel_2,
                             regressors[output])

model.fit([X, y0], Y, epochs=max_epochs, verbose=0, callbacks=[early_stop])

print("Efetuando predições com modelo 3")
predictions.append(predictOnPlotHorizon(model.get_weights()))

""" Plots e cálculo dos Sum of Squared Errors """

sse_models = []
model_no = 1

Y = np.array(Y_plot).reshape(X_plot.shape[0]*X_plot.shape[1], 1)
# Baseline tipo zero-order holder, calculada com mesmo horizonte de comparação
baseline = np.repeat(y0_plot[:, 0:1], repeats=horizon_plot, axis=0)

# Retornando à escala e desvio-padrão originais
baseline = Y_scaler.inverse_transform(baseline)
Y = Y_scaler.inverse_transform(Y)

baseline_mae = np.mean(np.abs(Y - baseline))
print("Baseline mean abs error = " + "%.5f" % baseline_mae)

# Remove uma dimensão, permitindo a plotagem. Isso é necessário porque as
# predições dos modelos parallel tem dimensão (1, timestep, exemplo).
# Com isso, fica a dimensão (1, timestep*exemplo).

for i in range(len(predictions)):
    predictions[i] = predictions[i].reshape(X_plot.shape[0]*X_plot.shape[1], 1)
    predictions[i] = Y_scaler.inverse_transform(predictions[i])
    print(f"Model {i+1} mean abs error = " + "%.5f"
          % (np.mean(np.abs(Y - predictions[i]))))

use("Qt5Agg")
plt.figure()
plt.plot(Y, linewidth=1.0, label='y')
plt.plot(baseline, linewidth=0.8, label='baseline')

for i in range(len(predictions)):
    plt.plot(predictions[i], linewidth=1.0, label=f"model {i+1}")

plt.xlabel("sample")
plt.title("Horizon = " + str(horizon_plot))
plt.legend()
plt.gcf().set_size_inches(17, 9.5)
manager = plt.get_current_fig_manager()
manager.window.showMaximized()
