import matplotlib.pyplot as plt
import data_utils as du
import rnn_utils as ru
import numpy as np


def single_plots(dictionary, raw_data, save=False, show_plots=True, save_folder="analysis", plot_baseline=True, horizon="default"):
    
    # Está função também plota o baseline e adiciona uma comparação no analysis dict
    sse_info = {}
    
    for y in dictionary:
        try:
            regressors = dictionary[y]["model"]["regressors"]
            
            K = dictionary[y]["model"]["K"]
            
            weights = dictionary[y]["model"]["weights"]
            
            if horizon == "default":
                horizon = dictionary[y]["K selection results"]["parameters"]["horizon"]

        except KeyError:
            print("Missing training results for " + y + ", output will not be plotted")
            continue
        
        try:
            dependency = dictionary[y]["dependency mask"]
        except KeyError:
            dependency = None
            
        data = du.trim_data(raw_data, y, dependency)
            
        X, Y, Y_scaler = du.build_sets_for_plotting(data, regressors, y)
        
        X, Y, y0 = ru.rnn_sets(X, Y, y, horizon, regressors[y])
        
        model = ru.rnn_model2(horizon=horizon, K=K)
        
        # Gambiarra só pra definir a input layer shape e ser possível setar os weights
        model.predict([X[0:1], y0[0:1]], verbose=False)
        model.set_weights(weights)
            
        predictions = model.predict([X, y0], verbose=False)

        # Remove uma dimensão, colocando as predições de volta em sequência
        predictions = predictions.reshape(X.shape[0]*X.shape[1], 1)
    
        Y = Y.reshape(X.shape[0]*X.shape[1], 1)

        model_sse = np.sum( np.square(Y - predictions) )
        
        baseline = np.repeat(y0[:, 0:1], repeats=horizon, axis=0)
            
        baseline_sse = np.sum( np.square(Y - baseline) )
        
        sse_info[y] = "Horizon: " + str(horizon) + ", model SSE: " + "%.5f" % model_sse + ", baseline: " + "%.5f" % baseline_sse
            
        baseline = Y_scaler.inverse_transform(baseline)
        
        predictions = Y_scaler.inverse_transform(predictions)
        
        Y = Y_scaler.inverse_transform(Y)

        plt.figure()
        plt.plot(Y, 'b', linewidth=1.2, label='y')
        plt.plot(predictions, 'r', linewidth=1.0, label='y_pred')
        
        if plot_baseline:
            plt.plot(baseline, 'y', linewidth=0.8, label='baseline')
            plt.title(y + ", horizon = " + str(horizon) + ", model SSE = " + "%.5f" % model_sse + ", baseline SSE = " + "%.5f" % baseline_sse)
        else:
            plt.title(y + ", horizon = " + str(horizon) + ", SSE = " + "%.5f" % model_sse)
            
        plt.xlabel("sample")
        plt.legend()
        
        # Maximiza o plot antes de mostrar/salvar. Funciona pro manager do Spyder
        plt.gcf().set_size_inches(17, 9.5)
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()
        
        if save:
            plt.savefig(save_folder + "\\figure_" + y, bbox_inches='tight') 
        if not show_plots:
            plt.close()
        
    return sse_info


def multiplots(dictionary, raw_data, size=3, save=False, save_folder="analysis", horizon="default"):
        
    plot_index = [0, 0]
    save_string = None
    
    for y in dictionary:
        try:
            regressors = dictionary[y]["model"]["regressors"]
            
            K = dictionary[y]["model"]["K"]
            
            weights = dictionary[y]["model"]["weights"]
            
            if horizon == "default":
                horizon = dictionary[y]["K selection results"]["parameters"]["horizon"]
            
        except KeyError:
            print("Missing training results for " + y + ", output will not be plotted")
            continue
        
        try:
            dependency = dictionary[y]["dependency mask"]
        except KeyError:
            dependency = None
                  
        data = du.trim_data(raw_data, y, dependency)
                       
        X, Y, Y_scaler = du.build_sets_for_plotting(data, regressors, y)
        
        X, Y, y0 = ru.rnn_sets(X, Y, y, horizon, regressors[y])    
        
        model = ru.rnn_model2(horizon=horizon, K=K)
        
        # Gambiarra só pra definir a input layer shape e ser possível setar os weights
        model.predict([X[0:1], y0[0:1]], verbose=False)
        model.set_weights(weights)
        
        predictions = model.predict([X, y0], verbose=False)
        
        predictions = predictions.reshape(X.shape[0]*X.shape[1], 1)
        
        predictions = Y_scaler.inverse_transform(predictions)
        
        Y = Y.reshape(X.shape[0]*X.shape[1], 1)
        
        model_sse = np.sum( np.square(Y - predictions) )
        
        Y = Y_scaler.inverse_transform(Y)
                                            
        if (plot_index == [0, 0]):
            fig, axs = plt.subplots(size, size)
            save_string = y
            
        axs[plot_index[0], plot_index[1]].plot(Y, 'b', linewidth=1, label=y)
        axs[plot_index[0], plot_index[1]].plot(predictions, 'r', linewidth=0.7, label=y + '_pred')
        axs[plot_index[0], plot_index[1]].legend()
        axs[plot_index[0], plot_index[1]].set_title(y + ", horizon = " + str(horizon) + ", SSE = " + "%.5f" % model_sse)
        
        # Se chegou no canto inferior direito
        if (plot_index == [size-1, size-1]):
            plot_index = [0, 0]
            # fig.tight_layout()
            fig.set_size_inches(17, 9.5)
            # Essa linha maximiza a janela do plot, funciona só se o manager for do tipo 'Qt5agg', 
            # que é o caso do Spyder. Para outros managers, o comando é diferente
            plt.get_current_fig_manager().window.showMaximized()
            if save:
                plt.savefig(save_folder + "\\multiplot_" + save_string + "_to_" + y, bbox_inches='tight')
        
        # Se chegou na borda da direita
        elif (plot_index[1] == size-1):
            plot_index[1] = 0
            plot_index[0] += 1
        else:
            plot_index[1] += 1
            
    # Save necessário de se fazer quando a divisão (size^2 / Ny) tem resto, pois nesse caso o último multiplot
    # não chega no canto inferior direito
    if (plot_index != [0, 0]):
         fig.set_size_inches(17, 9.5)
         plt.get_current_fig_manager().window.showMaximized()
         if save:
             plt.savefig(save_folder + "\\multiplot_" + save_string + "_to_" + y, bbox_inches='tight')
             
    return
             
            