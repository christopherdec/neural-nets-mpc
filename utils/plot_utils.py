# -*- coding: utf-8 -*-
""" Módulo de funções para plots

Utilizado após a criação dos modelos. Contém as funções
single_plots e multiplots.
"""

import matplotlib.pyplot as plt
from matplotlib import use
from .data_utils import build_sets, recursive_sets, trim_data
from .custom_models import ParallelModel, serial_parallel_model
import numpy as np

# Permite que plt.get_current_fig_manager() retorne um objeto "window", o qual
# pode ser maximizado automaticamente para ampliar o plot
use("Qt5Agg")


def single_plots(
        dictionary,
        raw_data,
        save=False,
        show_plots=True,
        save_folder="analysis",
        plot_baseline=True,
        horizon="default"):

    """Esta função faz um plot para cara modelo criado, a partir dos dados que
    estão no dicionário de treinamento. O horizonte de predição utilizado nos
    plots pode ser alterado. O desempenho é registrado num dicionário e a
    baseline de ordem 0 (constante) também pode ser plotada.

    Args:
        dictionary (dict): dicionário contendo info dos modelos criados
        raw_data (pd.DataFrame): conjunto de dados não trimmados
        save (bool, optional): salva as figuras. Defaults to False.
        show_plots (bool, optional): mostra os plots. Defaults to True.
        save_folder (str, optional): Defaults to "analysis".
        plot_baseline (bool, optional): Defaults to True.
        horizon (str, optional): "default" usa o mesmo horizonte utilziado
        no K selection. Defaults to "default".

    Returns:
        sse_info (dict): dicionário contendo o SSE (sum of squared errors) de
        cada modelo e sua baseline. é salvo depois no analysis_dict.
    """
    sse_info = {}

    for output in dictionary:
        try:
            regressors = dictionary[output]["model"]["regressors"]

            K = dictionary[output]["model"]["K"]

            weights = dictionary[output]["model"]["weights"]

            if horizon == "default":
                horizon = dictionary[output]["K selection \
                    results"]["parameters"]["horizon"]

        except KeyError:
            print(
                "Missing training results for " +
                output +
                ", output will not be plotted")
            continue

        try:
            dependency = dictionary[output]["dependency mask"]
        except KeyError:
            dependency = None

        data = trim_data(raw_data, output, dependency)

        X, Y, Y_scaler = build_sets(
            data, regressors, output, return_Y_scaler=True)

        if horizon > 1:
            X, Y, y0 = recursive_sets(
                X, Y, output, horizon, regressors[output])
            model = ParallelModel(horizon=horizon, K=K)
            baseline = np.repeat(y0[:, 0:1], repeats=horizon, axis=0)
        else:
            baseline = np.array(X[output + "(k-1)"])
            baseline = baseline.reshape(baseline.shape[0], 1)
            X = np.array(X)
            Y = np.array(Y)
            model = serial_parallel_model(K)

        # Força o custom model a definir a input layer shape, possibilitando
        # setar os weights
        if horizon > 1:
            model.predict([X[0:1], y0[0:1]], verbose=False)
        else:
            model.predict(X[0:1], verbose=False)

        model.set_weights(weights)

        if horizon > 1:
            predictions = model.predict([X, y0], verbose=False)
        else:
            predictions = model.predict(X, verbose=False)

        # Remove uma dimensão, colocando as predições de volta em sequência
        if horizon > 1:
            predictions = predictions.reshape(X.shape[0] * X.shape[1], 1)

            Y = Y.reshape(X.shape[0] * X.shape[1], 1)

        model_sse = np.sum(np.square(Y - predictions))

        baseline_sse = np.sum(np.square(Y - baseline))

        sse_info[output] = "Horizon: " + str(horizon) + ", model SSE: " + \
            "%.5f" % model_sse + ", baseline: " + "%.5f" % baseline_sse

        baseline = Y_scaler.inverse_transform(baseline)

        predictions = Y_scaler.inverse_transform(predictions)

        Y = Y_scaler.inverse_transform(Y)

        plt.figure()
        plt.plot(Y, 'b', linewidth=1.2, label='y')
        plt.plot(predictions, 'r', linewidth=1.0, label='y_pred')

        if plot_baseline:
            plt.plot(baseline, 'y', linewidth=0.8, label='baseline')
            plt.title(
                output + ", horizon = " + str(horizon) + ", model SSE = " +
                "%.5f" % model_sse + ", baseline SSE = " + "%.5f" %
                baseline_sse)
        else:
            plt.title(
                output + ", horizon = " + str(horizon) + ", SSE = " +
                "%.5f" % model_sse)

        plt.xlabel("sample")
        plt.legend()

        # Maximiza o plot antes de mostrar/salvar. Funciona pro manager do
        # Spyder
        plt.gcf().set_size_inches(17, 9.5)
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()

        if save:
            plt.savefig(save_folder + "\\figure_" +
                        output, bbox_inches='tight')
        if not show_plots:
            plt.close()

    return sse_info


def multiplots(
        dictionary,
        raw_data,
        size=[3, 3],
        save=False,
        save_folder="analysis",
        horizon="default",
        baseline_in_title=False):

    """Esta função faz vários plots por figura, facilitando a visualização no
    caso de muitas variáveis. Só plota modelos que foram criados, ou seja,
    cujas informações estiverem no training_dictionary. O horizonte de predição
    pode ser alterado.

    Args:
        dictionary (dict): dicionário contendo info dos modelos criados
        raw_data (pd.DataFrame): conjunto de dados não trimmados
        size (list): lista contendo altura e largura dos multiplots
        save (bool, optional): salva as figuras. Defaults to False.
        save_folder (str, optional): Defaults to "analysis".
        horizon (str, optional): "default" usa o mesmo horizonte utilziado
        no K selection. Defaults to "default".
        baseline_in_title (bool, optional): calcula a baseline e appenda no
        título. Defaults to False.

    """
    plot_index = [0, 0]
    save_string = None

    for output in dictionary:
        try:
            regressors = dictionary[output]["model"]["regressors"]

            K = dictionary[output]["model"]["K"]

            weights = dictionary[output]["model"]["weights"]

            if horizon == "default":
                horizon = dictionary[output]["K selection \
                    results"]["parameters"]["horizon"]

        except KeyError:
            print(
                "Missing training results for " +
                output +
                ", output will not be plotted")
            continue

        try:
            dependency = dictionary[output]["dependency mask"]
        except KeyError:
            dependency = None

        data = trim_data(raw_data, output, dependency)

        X, Y, Y_scaler = build_sets(
            data, regressors, output, return_Y_scaler=True)

        if horizon > 1:
            X, Y, y0 = recursive_sets(
                X, Y, output, horizon, regressors[output])
            model = ParallelModel(horizon=horizon, K=K)
        else:
            model = serial_parallel_model(K)
            X = np.array(X)
            Y = np.array(Y)

        # Força o custom model a definir a input layer shape, possibilitando
        # setar os weights
        if horizon > 1:
            model.predict([X[0:1], y0[0:1]], verbose=False)
        else:
            model.predict(X[0:1], verbose=False)
        model.set_weights(weights)

        if horizon > 1:
            predictions = model.predict([X, y0], verbose=False)
        else:
            predictions = model.predict(X, verbose=False)

        if horizon > 1:
            predictions = predictions.reshape(X.shape[0] * X.shape[1], 1)

            Y = Y.reshape(X.shape[0] * X.shape[1], 1)

        model_sse = np.sum(np.square(Y - predictions))

        predictions = Y_scaler.inverse_transform(predictions)

        if baseline_in_title:

            if horizon > 1:

                baseline = np.repeat(y0[:, 0:1], repeats=horizon, axis=0)

                baseline_sse = np.sum(np.square(Y - baseline))

        Y = Y_scaler.inverse_transform(Y)

        if (plot_index == [0, 0]):
            fig, axs = plt.subplots(size[0], size[1])
            save_string = output

        axs[plot_index[0], plot_index[1]].plot(
            Y, 'b', linewidth=1, label=output)
        axs[plot_index[0], plot_index[1]].plot(
            predictions, 'r', linewidth=0.7, label=output + '_pred')
        axs[plot_index[0], plot_index[1]].legend()

        if baseline_in_title:
            axs[plot_index[0], plot_index[1]].set_title(
                output + ", h = " + str(horizon) + ", SSE = " + "%.1f" %
                model_sse + ", base = " + "%.1f" % baseline_sse)
        else:
            axs[plot_index[0], plot_index[1]].set_title(
                output + ", horizon = " + str(horizon) + ", SSE = " + "%.2f"
                % model_sse)

        axs[plot_index[0], plot_index[1]].set_yticklabels([])
        axs[plot_index[0], plot_index[1]].set_xticklabels([])

        plt.gca().axes.get_yaxis().set_visible(False)
        plt.gca().axes.get_xaxis().set_visible(False)

        # Se chegou no canto inferior direito
        if (plot_index == [size[0] - 1, size[1] - 1]):
            plot_index = [0, 0]

            fig.tight_layout()
            fig.set_size_inches(17, 9.5)
            # Essa linha maximiza a janela do plot, funciona para o manager
            # 'Qt5agg', utilizado no Spyder.
            # Para outros managers, o comando é diferente.
            plt.get_current_fig_manager().window.showMaximized()
            if save:
                plt.savefig(
                    save_folder + "\\multiplot_" + save_string +
                    "_to_" + output, bbox_inches='tight')

        # Se chegou na borda da direita
        elif (plot_index[1] == size[1] - 1):
            plot_index[1] = 0
            plot_index[0] += 1
        else:
            plot_index[1] += 1

    # Save necessário de se fazer quando a divisão (size^2 / Ny) tem resto,
    # pois nesse caso o último multiplot não chega no canto inferior direito
    if (plot_index != [0, 0]):
        fig.set_size_inches(17, 9.5)
        plt.get_current_fig_manager().window.showMaximized()
        if save:
            plt.savefig(
                save_folder + "\\multiplot_" + save_string + "_to_" +
                output, bbox_inches='tight')

    return
