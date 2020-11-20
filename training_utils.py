import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import data_utils as du
import rnn_utils as ru
import time
import pandas as pd
    

def input_selection(data, output, params):
    
    start = time.time()
    
    max_stages = params["max stages"]
    trains_per_option = params["trains per option"]
    search_patience = params["search patience"]
    max_epochs = params["max epochs"]
    patience = params["early stop patience"]   
    resampling_factor = params["resampling factor"]
    K = params["hidden layer nodes"]   
    horizon = params["horizon"]
    starting_y_order = params["starting y order"]
    target_loss = params["target loss"]
    acceptable_loss = params["acceptable loss"]
    min_delta_loss = params["min delta loss"]
    # test_size = params["test size"]
    
    if resampling_factor > 1: data = data[::resampling_factor]
   
    search_results = { "parameters": params }
   
    early_stop = EarlyStopping(monitor='loss', patience=patience, verbose=1)
    
    stages_without_improvement = 0
          
    variables = data.columns
    
    no_variables = len(variables)
    
    regressors = pd.Series(data=np.zeros(no_variables), index=variables, dtype=int, name="ordem")
    
    if starting_y_order == 0:
        print("Starting y order must be greater than 0 for recursivity. Using order 1")
        starting_y_order = 1
        
    regressors[output] = starting_y_order

    best_loss = 999
    best_option_loss = 999
      
    for stage in range(max_stages):
        
        for option in variables:
            
            testing_regressors = regressors.copy(deep=True)

            testing_regressors[option] += 1
            
            print("Starting stage " + str(stage+1) + ", option = " + option + ", testing:")
            print(testing_regressors.to_string())
        
            # X_train, X_test, Y_train, Y_test = du.build_sets_for_input_selection(data, testing_regressors, output, test_size=test_size)
            
            X, Y = du.build_sets_for_input_selection(data, testing_regressors, output)
        
            X, Y, y0 = ru.rnn_sets(X, Y, output, horizon, testing_regressors[output])
        
            loss_sum_for_option = 0
            
            for train in range(trains_per_option):
                
                model = ru.LSTM_model(horizon, K) # ru.rnn_model2(horizon, K)
                
                history = model.fit([X, y0], Y, epochs=max_epochs, verbose=0, callbacks=[early_stop])
                
                # history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test) ...
        
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
        
        print("Best option: " + best_option + ", 'regressors' = " + str(regressors.to_numpy()) )
        
        search_results["Stage " + str(stage+1)] = "Best option: " + str(regressors.to_numpy()) + ", average loss: " + str(best_option_loss)

        # Verifica se o resultado da melhor opção é o melhor overall
        if (best_option_loss + min_delta_loss < best_loss):
            
            stages_without_improvement = 0
            best_regressors = regressors.copy(deep=True)
            best_loss = best_option_loss
            print("Best regressors updated")
            
        elif (best_loss < acceptable_loss):
            print("Best option doesn't improve performance and acceptable loss was already reached, interrupting search")
            stop_reason = "No significant improvement while acceptable loss"
            break
        else:
            print("Best option doesn't improve performance and acceptable loss was not reached, continuing search")
            stages_without_improvement += 1
        
        if (best_loss < target_loss):
            print("Target loss " + str(target_loss) + " reached, interrupting search")
            stop_reason = "target loss reached"
            break
        elif (stages_without_improvement == search_patience + 1):
            print("Interrupting search for not achieving improvements in " + str(search_patience + 1) + " stages")
            stop_reason = "stage patience run out"
            break
        elif (stage == max_stages - 1):
            print("Last stage completed")
            stop_reason = "Last stage completed"
            
    print("Best regressors found:")
    print(best_regressors.to_string())
            
 
    selected_regressors = []
            
    for variable in variables:
        
        # Dropa as variáveis com ordem 0
        if best_regressors[variable] == 0:
            best_regressors.drop(variable, inplace=True)
        else:
            selected_regressors.append(variable + " = " + str(best_regressors[variable]) )
    
    search_results["selected regressors"] = selected_regressors
    
    search_results["stop reason"] = stop_reason
       
    search_results["execution time"] = str( (time.time()-start)/60) + " minutes"
    
    print("Execution time: " + search_results["execution time"])
    
    return best_regressors, search_results


def K_selection(data, regressors, output, params):
    
    start = time.time()
        
    K_min = params["K min"]
    K_max = params["K max"]
    trains_per_K = params["trains per K"]
    search_patience = params["search patience"]
    max_epochs = params["max epochs"]
    resampling_factor = params["resampling factor"]
    patience = params["early stop patience"]
    horizon = params["horizon"]
    target_loss = params["target loss"]
    min_delta_loss = params["min delta loss"]
    #train_size = params["train size"]
    #val_size = params["train size"]  
    #test_size = params["train size"] 
    
    if resampling_factor > 1: data = data[::resampling_factor]
    
    search_results = { "parameters": params }
    
    best_loss = None
    best_loss_for_K = None
    stages_without_improvement = 0
    stop_reason = "K max reached" # default, para registro
    
    # X_train, X_val, X_test, Y_train, Y_val, Y_test = du.build_sets_for_K_selection(data, regressors, output, train_size, val_size, test_size)
    X, Y = du.build_sets_for_K_selection(data, regressors, output)
    
    X, Y, y0 = ru.rnn_sets(X, Y, output, horizon, regressors[output])
    
    early_stop = EarlyStopping(monitor='loss', patience=patience, verbose=1)

    for K in range (K_min, K_max + 1):
        
        print("Testing model with " + str(K) + " hidden nodes")
        
        for initialization in range (trains_per_K):
            
            model = ru.rnn_model2(horizon, K)
    
            history = model.fit([X, y0], Y, epochs=max_epochs, verbose=0, callbacks=[early_stop])
        
            # history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=epochs_max, verbose=0, callbacks=[early_stop])
            
            # loss_train_for_init = history.history['loss'][-1]
            # loss_val_for_init  = history.history['val_loss'][-1]
                    
            loss_for_init = history.history['loss'][-1]
            
            if (initialization == 0) or (loss_for_init < best_loss_for_K):
                best_loss_for_K = loss_for_init

                best_weights_for_K = model.get_weights()
                
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
            print("Stopping at K = " + str(K) + " for not achieving improvements in " + str(search_patience + 1) + " stages")
            stop_reason = "K patience run out"
            break
        elif (best_loss < target_loss):
            print("Stopping at K = " + str(K) + " for reaching target loss value")
            stop_reason = "target loss reached"
            break

    # print("Testando modelo obtido no conjunto de teste")
    # search_results["K = " + str(best_K)]["SSE test"] = best_model.evaluate(X_test, Y_test)
    
    search_results["stop reason"] = stop_reason
        
    search_results["execution time"] = str( (time.time()-start)/60) + " minutes"
    
    print("Execution time: " + search_results["execution time"])
               
    return best_K, best_weights, search_results


"""
Começa com todas as variáveis tendo a mesma ordem inicial
Vai diminuindo -1 até chegar na com melhor resultado
"""
def input_selection2(data, output, params):
    
    start = time.time()
    
    target_loss = params["target loss"]
    acceptable_loss = params["acceptable loss"]
    min_delta_loss = params["min delta loss"]
    
    max_stages = params["max stages"]

    trains_per_option = params["trains per option"]
    
    search_patience = params["search patience"]
    
    max_epochs = params["max epochs"]
    
    patience = params["early stop patience"]  
    
    resampling_factor = params["resampling factor"]
    
    K = params["hidden layer nodes"] 
    
    horizon = params["horizon"]
    
    starting_order = params["starting y order"] # starting y order aqui é usado como a ordem inicial de todas variáveis
    
    if resampling_factor > 1: data = data[::resampling_factor]
   
    search_results = { "parameters": params }
   
    early_stop = EarlyStopping(monitor='loss', patience=patience, verbose=1)
    best_option = None
    best_option_loss = None
    
    stages_without_improvement = 0
    
    stop_reason = "Last stage completed" # default, para registro
          
    variables = data.columns
    
    no_variables = len(variables)
    
    if starting_order == 0:
        print("Starting variables order must be greater than 0, using default 1")
        starting_order = 1
        
    regressors = pd.Series(data=starting_order*np.ones(no_variables), index=variables, dtype=int, name="ordem")
        
    """ Calcula os resultados obtidos com o 'regressors' inicial """
    
    X, Y = du.build_sets_for_input_selection(data, regressors, output)
    X, Y, y0 = ru.rnn_sets(X, Y, output, horizon, regressors[output])
    
    loss_sum_for_option = 0
    
    for train in range(trains_per_option):
    
        model = ru.rnn_model2(horizon, K)
        history = model.fit([X, y0], Y, epochs=max_epochs, verbose=0, callbacks=[early_stop])
        
        loss_sum_for_option += history.history['loss'][-1]
    
    best_regressors = regressors.copy(deep=True)
    best_loss = loss_sum_for_option / trains_per_option
    
    print("Initial best loss = " + str(best_loss))
      
    for stage in range(max_stages):
        
        for option in variables:
            
            testing_regressors = regressors.copy(deep=True)
            
            # se a ordem iria para -1, ignora a opção e passa para próxima
            if testing_regressors[option] -1 < 0:
                continue
            # se a ordem da saida iria para 0, ignora e passa, pois tem que haver recursividade
            elif regressors.index[option] == output and testing_regressors[option] -1 < 1:
                continue
            
            # EMPORÁRIO SÓ PRA TESTAR UMA COISA
            elif regressors.index[option] == output:
                continue
           
            else:
                testing_regressors[option] -= 1
            
            
            print("Starting stage " + str(stage+1) + ", option " + str(option+1) + ", testing:")
            
            print(testing_regressors.to_string())
            
            X, Y = du.build_sets_for_input_selection(data, testing_regressors, output)
        
            X, Y, y0 = ru.rnn_sets(X, Y, output, horizon, testing_regressors[output])
        
            loss_sum_for_option = 0
            
            for train in range(trains_per_option):
                
                model = ru.rnn_model2(horizon, K)
                
                history = model.fit([X, y0], Y, epochs=max_epochs, verbose=0, callbacks=[early_stop])
                          
                loss_sum_for_option += history.history['loss'][-1]
                
            # Calcula a média do loss para a opção
            loss_for_option = loss_sum_for_option / trains_per_option
            
            print("Average loss for option = " + str(loss_for_option))
            
            if (option == 0) or (loss_for_option < best_option_loss):
                
                best_option = option
                best_option_loss = loss_for_option
                best_option_regressors = testing_regressors.copy(deep=True)

        # Ao sair do loop, pega o melhor regressors dentre as opções
        regressors = best_option_regressors.copy(deep=True)
        
        print("Best option: " + str(best_option+1) + ", 'regressors' = " + str(regressors.to_numpy()) )
        
        search_results["Stage " + str(stage+1)] = "Best option: " + str(regressors.to_numpy()) + ", average loss: " + str(best_option_loss)

        # Verifica se o resultado da melhor opção é o melhor overall
        if (best_option_loss + min_delta_loss < best_loss):
            
            stages_without_improvement = 0
            best_regressors = regressors.copy(deep=True)
            best_loss = best_option_loss
            print("Best 'regressors' updated")
            
        elif (best_loss < acceptable_loss):
            print("Best option doesn't improve performance and acceptable loss was already reached, interrupting search")
            stop_reason = "No significant improvement while acceptable loss"
            break
        else:
            print("Best option doesn't improve performance and acceptable loss was not reached, continuing search")
            stages_without_improvement += 1
        
        if (best_loss < target_loss):
            print("Target loss " + str(target_loss) + " reached, interrupting search")
            stop_reason = "target loss reached"
            break
        elif (stages_without_improvement == search_patience + 1):
            print("Interrupting search for not achieving improvements in " + str(search_patience + 1) + " stages")
            stop_reason = "stage patience run out"
            break
        
        # Se chegou num regressors como [0, 0, 0, 0, 0, 1], não tem mais o que testar. Finaliza a procura.
        elif sum(regressors) == 1:
            print("Interrupting search for reaching only one input")
            stop_reason = "tested all possibilities from starting order"
            break
        
            
    print("Best regressors found:")
    print(best_regressors.to_string())
            
 
    selected_regressors = []
            
    for variable in variables:
        
        # Dropa as variáveis com ordem 0
        if best_regressors[variable] == 0:
            best_regressors.drop(variable, inplace=True)
        else:
            selected_regressors.append(variable + " = " + str(best_regressors[variable]) )
    
    search_results["selected regressors"] = selected_regressors
    
    search_results["stop reason"] = stop_reason
       
    search_results["execution time"] = str( (time.time()-start)/60) + " minutes"
    
    print("Execution time: " + search_results["execution time"])
    
    return best_regressors, search_results

