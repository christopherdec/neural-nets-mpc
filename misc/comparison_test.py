from utils import data_utils as du
from utils import custom_models as cm
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
import scipy.io
import matplotlib.pyplot as plt
import datetime

# rm -rf ./logs/
tf.keras.backend.clear_session()
tf.random.set_seed(0)
tf.autograph.set_verbosity(0)
# tf.get_logger().setLevel('INFO')
tf.autograph.experimental.do_not_convert

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus),
                  "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

output = "y1"
raw_data, variable_names, _ = du.load_data("batch_1")
data = du.trim_data(raw_data, output)
regressors = pd.Series(data=np.array([2, 2, 2, 2, 2, 2]),
                       index=data.columns,
                       dtype=int,
                       name="order")

X_orig, Y_orig, Y_scaler = du.build_sets(data,
                                         regressors,
                                         output,
                                         return_Y_scaler=True,
                                         scaler="Standard")

# sklearn.preprocessing.StandardScaler

# regressors_teste = pd.Series(data=np.array([3, 3, 3, 3, 3]),
#                              index=data.columns,
#                              dtype=int,
#                              name="order")
# X_teste, Y_teste = du.build_sets(data, regressors_teste, output)

correlation = pd.concat([X_orig, Y_orig], axis=1).corr()

# df = pd.concat([X_orig, Y_orig])
# correlation = df.corr()

horizon_model_parallel_1 = 5
horizon_model_parallel_2 = 20
horizon_plot = 100

# testar no optimizer adam: epsilon 1.0 e 0.1, o default é 1e-07
# testar sgd: momentum 0.5, 0.8, o default é 0

optimizer = tf.keras.optimizers.Adam()
loss = 'mse'
# loss = cm.SSE()

K = 8
max_epochs = 1

early_stop = EarlyStopping(monitor='loss', patience=3, verbose=1)
# early_stop = EarlyStopping(monitor='val_loss', patience=3, verbose=1,
#                            mode='min', restore_best_weights=True)

predictions = []

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
plot_model.predict([X_plot[0:1], y0_plot[0:1]], verbose=False)


def predictOnPlotHorizon(weights):
    plot_model.set_weights(weights)
    return plot_model.predict([X_plot, y0_plot], verbose=False)


""" MODELO SERIAL-PARALLEL """

model = cm.serial_parallel_model(K, optimizer=optimizer, loss=loss)

model.fit(np.array(X_orig.copy(deep=True)),
          np.array(Y_orig.copy(deep=True)),
          epochs=max_epochs,
          verbose=0,
          callbacks=[early_stop])

predictions.append(predictOnPlotHorizon(model.get_weights()))

""" MODELO PARALLEL 1 """

model = cm.ParallelModel(horizon=horizon_model_parallel_1,
                         K=K,
                         optimizer=optimizer,
                         loss=loss)
# model =  cm.ParallelModel2(regressors,
#                            output,
#                            horizon_model_parallel_1,
#                            K,
#                            optimizer=optimizer,
#                            loss=loss)

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

predictions.append(predictOnPlotHorizon(model.get_weights()))

# tensorboard --logdir logs/fit
# tf.keras.utils.plot_model(model)

""" MODELO PARALLEL 2 """

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

predictions.append(predictOnPlotHorizon(model.get_weights()))

""" Plots """

sse_models = []
model_no = 1

Y = np.array(Y_plot).reshape(X_plot.shape[0]*X_plot.shape[1], 1)
baseline = np.repeat(y0_plot[:, 0:1], repeats=horizon_plot, axis=0)

# Retornando à escala e desvios originais
baseline = Y_scaler.inverse_transform(baseline)
Y = Y_scaler.inverse_transform(Y)

baseline_mae = np.mean(np.abs(Y - baseline))
print("Baseline mean abs error = " + "%.5f" % baseline_mae)

# Remove uma dimensão, colocando as predições de volta em sequência
for i in range(len(predictions)):
    predictions[i] = predictions[i].reshape(X_plot.shape[0]*X_plot.shape[1], 1)
    predictions[i] = Y_scaler.inverse_transform(predictions[i])
    print(f"Model {i+1} mean abs error = " + "%.5f"
          % (np.mean(np.abs(Y - predictions[i]))))

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
