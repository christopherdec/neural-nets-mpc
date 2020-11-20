import numpy as np
import pickle
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler      
import scipy.io
import pandas as pd
import h5py


# Esse é o lote com dados reais do processo para efetuar o treinamento das redes
# A Carolina me enviou estes dados no formato .mat, contém 99.1k de amostras
def load_lote_1():
      
    input_ = scipy.io.loadmat("dados\\lote 1 treinamento\\input.mat")
    output = scipy.io.loadmat("dados\\lote 1 treinamento\\output.mat")
    
    inputs = []
    outputs = []
    variable_names = {}
    columns = []
    
    u, y = 1, 1
    
    for key, value in input_.items():
        if type(value) == np.ndarray:
            inputs.append(value.flatten())
            variable_names["u" + str(u)] = key
            columns.append("u" + str(u))
            u += 1
       
    inputs = np.transpose(np.array(inputs))
    
    for key, value in output.items():
        if type(value) == np.ndarray:
            outputs.append(value.flatten())
            variable_names["y" + str(y)] = key
            columns.append("y" + str(y))
            y += 1
         
    # Ny = len(outputs)
    
    outputs = np.transpose(np.array(outputs))
    
    raw_data = np.hstack((inputs, outputs))
    
    raw_data = pd.DataFrame(raw_data, columns=columns)
    
    return raw_data, variable_names


# Outro lote com dados reais do processo, sendo utilizado para avaliar os modelos criados no arquivo predict.py
# A Carolina me enviou estes dados no formato .h5, contém 205.3k de amostras
def load_lote_2():

    inputs = pd.read_hdf("dados\\lote 2 avaliacao\\input_validation.h5")

    outputs = pd.read_hdf("dados\\lote 2 avaliacao\\output_validation.h5")
    
    raw_data = pd.concat([inputs, outputs], axis=1)
    
    raw_data.reset_index(drop=True, inplace=True)
    
    variable_names = [ list(raw_data.columns), [] ]
    
    for i in range (1, inputs.shape[1] + 1):
        variable_names[1].append("u" + str(i))
        
    for i in range (1, outputs.shape[1] + 1):
        variable_names[1].append("y" + str(i))    
        
    raw_data.columns = variable_names[1]
    
    return raw_data, variable_names

def load_lote_3():
    
    inputs = pd.read_hdf("dados\\lote 3 teste rnn\\inputs.h5").dropna()

    outputs = pd.read_hdf("dados\\lote 3 teste rnn\\outputs.h5")
    
    outputs.drop( outputs.head(2).index, inplace=True)
            
    raw_data = pd.concat([inputs, outputs], axis=1)
    
    raw_data.reset_index(drop=True, inplace=True)
    
    variable_names = [list(raw_data.columns)]
    
    variable_names.append(['u1', 'u2', 'u3', 'u4', 'u5', 'u6', 'u7', 'y1', 'y2', 'y3', 'y4'])
    
    raw_data.columns = variable_names[1]
    
    return raw_data, variable_names


def load_pickle(path):
    try:
        f = open(path, "rb")
    except IOError:
        print("'" + str(path) + "'" + " was not found, returning an empty object")
        return {}
    else:
        stuff = pickle.load(f)
        f.close()
    return stuff
    
def save_pickle(stuff, path):
    pickle.dump( stuff, open(path, "wb"))
  
    
"""
trim_data retorna o conjunto de dados apenas com as variáveis que serão utilizadas na rede neural da saída informada
'dependency' é uma lista que indica quais variáveis influenciam na saida informada
Para casos em que não se sabe quais são estas variáveis, 'dependency' não é informada e assume-se que a saída 
é influenciada apenas pelas variáveis manipuladas 'u' e por ela mesma. 
"""
def trim_data(raw_data, output, dependency=None):
    
    if dependency is None:
        
        variaveis_influentes = [col for col in raw_data if col.startswith('u')]

        variaveis_influentes.append(output)
        
        data = raw_data[variaveis_influentes]

    else:       
        data = raw_data[dependency]
        
    return data


# Atualmente, essa função deve ser chamada no console para adicionar a lista de dependencias de uma variavel
# 'dependency' deve ser uma lista com as variáveis que influenciam na saida informada
# Exemplo: dependency = ['u1', 'u4', 'y1', 'y3', 'y7']
def add_dependency(output, dependency):
    
    dictionary = load_pickle("analysis\\dictionary.pickle")
    
    try:
        dictionary[output]["dependency mask"] = dependency
    except KeyError:
        print("No entry found for " + output + " on the dictionary, creating an empty object")
        dictionary[output] = { "dependency mask" : dependency }
        
    save_pickle(dictionary, "analysis\\dictionary.pickle")
    
    print ("Added dependency = " + str(dependency) + " for output " + output)


""" Funções de processamento de dados, utilizadas durante treinamentos e plots """
     
def build_sets_for_input_selection(data, regressors, output): # , test_size):
    
    X, Y = build_sets(data, regressors, output)
    
    X[X.columns] = MinMaxScaler().fit_transform(X[X.columns])
    Y[Y.columns] = MinMaxScaler().fit_transform(Y[Y.columns])
    
    # return train_test_split(X, Y, test_size=test_size, shuffle=True, random_state=42)

    return X, Y       
    

def build_sets_for_K_selection(data, regressors, output): # train_size, val_size, test_size):
            
    X, Y = build_sets(data, regressors, output)
    
    X[X.columns] = MinMaxScaler().fit_transform(X[X.columns])
    Y[Y.columns] = MinMaxScaler().fit_transform(Y[Y.columns])
        
    #X_train, X_val, Y_train, Y_val = train_test_split(X, Y, train_size=train_size, shuffle=True, random_state=42)
        
    #X_val, X_test, Y_val, Y_test = train_test_split(X_val, Y_val, test_size=test_size/(test_size + val_size), shuffle=False)
        
    #return X_train, X_val, X_test, Y_train, Y_val, Y_test
    
    return X, Y


def build_sets_for_plotting(data, regressors, output):
    
    X, Y = build_sets(data, regressors, output)
        
    X[X.columns] = MinMaxScaler().fit_transform(X[X.columns])    
    Y_scaler = MinMaxScaler().fit(Y)    
    Y[Y.columns] = Y_scaler.transform(Y[Y.columns])
    
    return X , Y, Y_scaler


def build_sets(data, regressors, output):
                   
    maior_ordem = max(regressors)
                  
    X = pd.DataFrame()
    
    Y = data[output].copy().shift(-maior_ordem).dropna().reset_index(drop=True).rename(output + "(k)").to_frame()    
    
    for variavel, ordem in regressors.items():
                             
        for atraso in range(1, ordem + 1):
            
            dados_variavel = data[variavel].copy().shift(atraso - maior_ordem).dropna().reset_index(drop=True)
                                                        
            X[variavel + "(k-" + str(atraso) + ")"] = dados_variavel.drop(dados_variavel.tail(1).index)
                            
    return X, Y