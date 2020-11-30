import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
import sklearn.utils
import scipy.io
import pandas as pd
import math

# Função utilizada para carregar os dados de treinamento
def load_data(batch="batch_1"):
    
    # Esse é o primeiro lote que a Carolina me enviou
    # Contém dados reais do processo para efetuar o treinamento das redes
    # Contém 99.1k amostras
    # Tem 5 variáveis 'u' e 12 variáveis 'y'
    if batch == "batch_1":
      
        input_ = scipy.io.loadmat("data\\batch_1\\input.mat")
        output = scipy.io.loadmat("data\\batch_1\\output.mat")
        
        raw_data = []
        variable_names = {}
        
        u, y = 1, 1
        
        for key, value in input_.items():
            if type(value) == np.ndarray:
                raw_data.append(value.flatten())
                variable_names["u" + str(u)] = key
                u += 1
             
        for key, value in output.items():
            if type(value) == np.ndarray:
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

        inputs = pd.read_hdf("data\\batch_2\\input_validation.h5")
    
        outputs = pd.read_hdf("data\\batch_2\\output_validation.h5")
        
        raw_data = pd.concat([inputs, outputs], axis=1)
        
        raw_data.reset_index(drop=True, inplace=True)
        
        variable_names = [ list(raw_data.columns), [] ]
        
        for i in range (1, inputs.shape[1] + 1):
            variable_names[1].append("u" + str(i))
            
        for i in range (1, outputs.shape[1] + 1):
            variable_names[1].append("y" + str(i))    
            
        raw_data.columns = variable_names[1]
        
        Ny = 12


    # Terceiro lote com dados reais do processo
    # Tem 7 variáveis 'u' e 4 variáveis 'y'
    # Requer uma etapa de pre-processamento para tirar os valores nan e sincronizar as amostras
    # Contém 99.1k amostras
    elif batch == "batch_3":
    
        inputs = pd.read_hdf("data\\batch_3\\inputs.h5").dropna()
    
        outputs = pd.read_hdf("data\\batch_3\\outputs.h5")
        
        outputs.drop( outputs.head(2).index, inplace=True)
                
        raw_data = pd.concat([inputs, outputs], axis=1)
        
        raw_data.reset_index(drop=True, inplace=True)
        
        variable_names = [list(raw_data.columns)]
        
        variable_names.append(['u1', 'u2', 'u3', 'u4', 'u5', 'u6', 'u7', 'y1', 'y2', 'y3', 'y4'])
        
        raw_data.columns = variable_names[1]
        
        Ny = 4
        
    # Dados do sistema de testes feito no Simulink com o intuito de validar os algoritmos
    # Contém 10k amostras
    # Regressors utilizado para gerar os dados = [u1 = 1, u2 = 0, u3 = 0, u4 = 2, y1 = 3]
    elif batch == "simulink":

        raw_data = scipy.io.loadmat('data\\simulink\\sistema_teste.mat')['simout']
        
        raw_data = pd.DataFrame(data=raw_data, columns=['u1', 'u2', 'u3', 'u4', 'y1'])
        
        variable_names, Ny = None, 1


    # Dados do simluador do processo, utilizados na iteração anterior do projeto
    # Foram levantados no software EMSO
    # Contém 18k amostras
    elif batch == "emso":
        
        raw_data = np.transpose( np.array( load_pickle("data\\emso_simulator\\dados18k_p74.pickle") ) )
        
        columns = load_pickle("data/emso_simulator/columns.pickle")
        
        raw_data = pd.DataFrame(data=raw_data, columns=columns)
        
        variable_names = load_pickle("data/emso_simulator/variable_names.pickle")
        
        Ny = 36
    
       
    return raw_data, variable_names , Ny
        
        
# Função utilizada para carregar arquivos .pickle
def load_pickle(path):
    try:
        f = open(path, "rb")
    except IOError:
        stuff = {}
    else:
        stuff = pickle.load(f)
        f.close()
    return stuff
   
# Função utilizada para salvar arquivos .pickle 
def save_pickle(stuff, path):
    pickle.dump( stuff, open(path, "wb"))
  
 
"""
Função para informar a dependência de uma variável. Atualmente, deve ser chamada no console
'dependency' deve ser uma lista com as variáveis que influenciam na saida informada
Exemplo de parâmetros: output = 'y1', dependency = ['u1', 'u4', 'y1', 'y3', 'y7'], 
"""
def add_dependency(output, dependency):
    
    training_dictionary = load_pickle("analysis\\dictionary.pickle")
    
    try:
        training_dictionary[output]["dependency mask"] = dependency
    except KeyError:
        print("No entry found for " + output + " on the dictionary, creating an empty object")
        training_dictionary[output] = { "dependency mask" : dependency }
        
    save_pickle(training_dictionary, "analysis\\dictionary.pickle")
    
    print ("Added dependency = " + str(dependency) + " for output " + output)
   
"""
trim_data retorna o conjunto de dados apenas com as variáveis que serão utilizadas na rede neural da saída atual
'dependency' é uma lista que indica essas variáveis
Para casos em que não se sabe quais são estas variáveis, 'dependency' não é informada e assume-se que a saída 
é influenciada apenas pelas variáveis manipuladas 'u' e por ela mesma
"""
def trim_data(raw_data, output, dependency=None):
    
    if dependency is None:
        
        variaveis_influentes = [col for col in raw_data if col.startswith('u')]

        variaveis_influentes.append(output)
        
        data = raw_data[variaveis_influentes]

    else:       
        data = raw_data[dependency]
        
    return data


"""
A função build_sets extrai o máximo de exemplos possíveis dos dados, levando em conta a variável que tiver a
maior ordem no 'regressors'.
Exemplo: se a maior ordem é 3, o primeiro exemplo é o da terceira amostra
regressors = [u1 = 1, u2 = 2, y1 = 3]
y1(k) = f(u1(k-1), u2(k-1), u2(k-2), y1(k-1), y1(k-2), y1(k-3))
"""
def build_sets(data, regressors, output, return_Y_scaler=False):
                   
    maior_ordem = max(regressors)
                  
    X = pd.DataFrame()
    
    Y = data[output].copy().shift(-maior_ordem).dropna().reset_index(drop=True).rename(output + "(k)").to_frame()    
    
    for variavel, ordem in regressors.items():
                             
        for atraso in range(1, ordem + 1):
            
            dados_variavel = data[variavel].copy().shift(atraso - maior_ordem).dropna().reset_index(drop=True)
                                                        
            X[variavel + "(k-" + str(atraso) + ")"] = dados_variavel.drop(dados_variavel.tail(1).index)
             
    # Normaliza os conjuntos
    X[X.columns] = MinMaxScaler().fit_transform(X[X.columns])    
    Y_scaler = MinMaxScaler().fit(Y)    
    Y[Y.columns] = Y_scaler.transform(Y[Y.columns])
    
    if return_Y_scaler: 
        return X, Y, Y_scaler 
    else: 
        return X, Y   
    
"""
Essa função permite efetuar treinamentos na configração Parallel.
Exemplo: se existem 10k dados e o horizonte de predição é 50, serão 200 exemplos de 50 amostras em sequência
Cada amostra em sequencia é chamado de uma janela. 
Se a divisão não for exata, são removidas as últimas amostras até a conta fechar. Não tem problema em fazer isso, pois
geralmente são bastante amostras.
O vetor y0 contém os estados iniciais dos regressores da saída, para cada janela.
"""
def recursive_sets(X, Y, output, horizon, y_order, shuffle=False):
        
    # len(x) seria igual ao número de janelas para um horizon = 1
    num_janelas = math.floor(len(X)/horizon)
        
    # Valores a remover para que cada janela tenha exatamente o mesmo número de exemplos
    # Essa é uma estratégica melhor do que repetir o último valor
    remocoes = int((len(X)/horizon)*horizon - num_janelas * horizon)   
            
    X.drop(X.tail(remocoes).index, inplace=True)       
    Y.drop(X.tail(remocoes).index, inplace=True)
        
    # Para X e Y, coloca as amostras numa lista com 'num_janelas' exemplos de sequências de 'horizon' amostragens
    X = np.split(X, num_janelas)
    Y = np.split(Y, num_janelas)
    
    if shuffle: X, Y = shuffle_sets(X, Y)
     
    # y0 contém os valores de y(k-1)...(k-y_order)
    # Existe uma linha em y0 para cada num_janelas
    # 'horizon' não participa das contas, pois os valores de y0 só são utilizados no primeiro instante, como estado inicial
    y0 = np.zeros((num_janelas, y_order))
    
    # Preenche y0 a partir dos regressores de y1 contidos em X
    for i in range(num_janelas):
        for j in range(y_order):  
            y0[i, j] = X[i].head(1)[output + "(k-" + str(j+1) + ')']
        
        # Agora os regressores de y1 em X devem ser removidos, pois estes serão informados por y0
    for janela in X:
        for i in range(y_order):
            janela.drop(output + "(k-" + str(i+1) + ')', inplace=True, axis=1)
                
    return np.array(X), np.array(Y), y0


def shuffle_sets(X, Y):
    
    index_shuf = list(range(len(X)))
    
    index_shuf = sklearn.utils.shuffle(index_shuf, random_state=42)
    
    X = [X[i] for i in index_shuf] 
    
    Y = [Y[i] for i in index_shuf]
    
    return X, Y
    
