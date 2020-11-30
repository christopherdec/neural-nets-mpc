import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
import data_utils as du
import custom_models as cm
import time
import pandas as pd
from sklearn.model_selection import train_test_split 
    

def input_selection(data, output, params):
    
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
    starting_y_order = params["starting y order"]
    target_loss = params["target loss"]
    acceptable_loss = params["acceptable loss"]
    min_delta_loss = params["min delta loss"]
    validation_size = params["validation size"]   
    structure = params["structure"]
    
    if partition_size < 1: 
        data, _ = np.split(data, [int(partition_size*len(data))])
   
    search_results = { "parameters": params } # grava os parâmetros que foram utilizados
   
    # o early stop é um callback do tf que para o treino assim que é detectado overfitting
    early_stop = EarlyStopping(monitor='loss', patience=patience, verbose=1)
    
    stages_without_improvement = 0
          
    variables = data.columns
    
    regressors = pd.Series(data=np.zeros(len(variables)), index=variables, dtype=int, name="ordem")
    
    if starting_y_order == 0: 
        print("Starting y order must be greater than 0 for recursivity. Using order 1")
        starting_y_order = 1
        
    regressors[output] = starting_y_order

    # inicia os placeholders em valores altos para serem substituídos no primeiro estágio
    best_loss = 999
    best_option_loss = 999
      
    for stage in range(max_stages):
        
        # cada variável representa uma opção, no caso a opção de incrementar a respectiva ordem em 1
        for option in variables:
            
            testing_regressors = regressors.copy(deep=True)

            testing_regressors[option] += 1
            
            print("Starting stage " + str(stage+1) + ", option = " + option + ", testing:")
            print(testing_regressors.to_string())
        
            X, Y = du.build_sets(data, testing_regressors, output)
        
            if horizon > 1:
                X, Y, y0 = du.recursive_sets(X, Y, output, horizon, testing_regressors[output], shuffle=True)
            else:
                X, Y = du.shuffle_sets(np.array(X), np.array(Y))
        
            loss_sum_for_option = 0
            
            # faz varios treinos e tira a média, para ter resultados mais consistentes
            for train in range(trains_per_option):
                
                if horizon > 1:
                    
                    if structure == "DLP":
                        model = cm.rnn_model(horizon, K)
                    elif structure == "LSTM":
                        model = cm.LSTM_model(horizon, K)
                        
                    history = model.fit([X, y0], Y, epochs=max_epochs, validation_split=validation_size, verbose=0, callbacks=[early_stop])
                
                else:
                    model = cm.sp_model(K)
                    history = model.fit(X, Y, epochs=max_epochs, validation_split=validation_size, shuffle=True, verbose=0, callbacks=[early_stop])
                
                
                if validation_size > 0: 
                    loss_for_training  = history.history['val_loss'][-1]
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
            
    # Dropa as variáveis com ordem 0 e preenche a lista selected_regressors para incluir no dicionário
    for variable in variables:
        
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
    partition_size = params["partition size"]
    patience = params["early stop patience"]
    horizon = params["horizon"]
    target_loss = params["target loss"]
    min_delta_loss = params["min delta loss"]
    validation_size = params["validation size"]
    test_size = params["test size"] 
    
    if partition_size < 1: 
        data, _ = np.split(data, [int(partition_size*len(data))])
    
    search_results = { "parameters": params } # grava os parâmetros que foram utilizados
   
    # o early stop é um callback do tf que para o treino assim que é detectado overfitting
    early_stop = EarlyStopping(monitor='loss', patience=patience, verbose=1)
    
    best_loss = None
    best_loss_for_K = None
    stages_without_improvement = 0
    stop_reason = "K max reached" # default, para registro
    
    X, Y = du.build_sets(data, regressors, output)
    
    if horizon > 1:
        X, Y, y0 = du.recursive_sets(X, Y, output, horizon, regressors[output], shuffle=True)
    else:
        X, Y = du.shuffle_sets(np.array(X), np.array(Y))
    
    """
    Essa chamada faz o split do test set, para ser utilizada ao final para avaliação do melhor modelo obtido.
    O split de treino/validação é feito pelo método model.fit quando os modelos são treinados.
    Como shuffle já foi feito na recursive_sets, aqui é necessário somente realizar a partição mesmo.
    """
    if horizon > 1 and test_size is not False:
        X, X_test, Y, Y_test, y0, y0_test = train_test_split(X, Y, y0, test_size=test_size, random_state=42)
    elif test_size is not False:
        X, X_test, Y, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)
        
    # a busca começa em K_min e vai até K_max, onde K é o número de neurônios na camada oculta
    for K in range (K_min, K_max + 1):
        
        print("Testing model with " + str(K) + " hidden nodes")
        
        # para ter resultados mais consistentes, cada K é inicializado algumas vezes, só a melhor inicialização é considerada
        for initialization in range (trains_per_K):
            
            if horizon > 1:
                model = cm.rnn_model(horizon, K)
                history = model.fit([X, y0], Y, validation_split=validation_size, epochs=max_epochs, verbose=0, callbacks=[early_stop])
            else:
                model = cm.sp_model(K)
                history = model.fit(X, Y, validation_split=validation_size, shuffle=True, epochs=max_epochs, verbose=0, callbacks=[early_stop])
    
            
            
            loss_for_init  = history.history['val_loss'][-1]            
                    
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
            print("Stopping at K = " + str(K) + " for not achieving improvements in " + str(search_patience + 1) + " stages")
            stop_reason = "K patience run out"
            break
        elif (best_loss < target_loss):
            print("Stopping at K = " + str(K) + " for reaching target loss value")
            stop_reason = "target loss reached"
            break


    if test_size is not False:
        print("Testando modelo obtido no conjunto de teste")
        model = cm.rnn_model(horizon, best_K)
        # Força o custom model a definir a input layer shape, possibilitando setar os weights
        model.predict([X[0:1], y0[0:1]], verbose=False)
        model.set_weights(best_weights)
        search_results["Loss in test set"] = model.evaluate([X_test, y0_test], Y_test)
    
    search_results["best K"] = best_K
    
    search_results["stop reason"] = stop_reason
        
    search_results["execution time"] = str( (time.time()-start)/60) + " minutes"
    
    print("Execution time: " + search_results["execution time"])
               
    return best_K, best_weights, search_results
