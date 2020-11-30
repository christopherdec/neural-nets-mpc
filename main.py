import data_utils as du
import training_utils as tu
import analysis_utils as au
import plot_utils as pu
import tensorflow as tf

# Diminui a verbosidade do tensorflow
import logging, os 
logging.disable(logging.WARNING) 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Reseta todo estado gerado pelo tensorflow e define a seed para reproducibilidade
tf.keras.backend.clear_session()
tf.random.set_seed(0)
tf.autograph.set_verbosity(0)

""" Carregamento de dados """
# Opções atuais: batch_1, batch_2, batch_3, simulink, emso
# Para novos datasets, uma novo opção deve ser adicionada na função load_data
# Lá está explicado a origem desses conjuntos também
raw_data, variable_names, Ny = du.load_data("batch_1")

# O dicionário de treinamento contém informações que vão adicionadas ao longo da criação dos modelos
training_dictionary = du.load_pickle("analysis\\dictionary.pickle")

# O dicionário de análise contém informações sintetizadas a partir do dicionário de treinamento
analysis_dict = du.load_pickle("analysis\\analysis_dict.pickle")

# Para explorar os dicionários, utilizar o Variable explorer do Spyder ou o variableInspector do Jupyter Lab

""" Configurações de execução """

# Se "run":False, o loop de treinamentos é pulado, indo direto para análise e plots dos modelos que já foram criados
exec_cfg = {
    "run" : False,
    "first y" : 1, # Permite escolher pra quais saídas serão criados modelos
    "last y" : Ny,
    "find inputs" : True, # Só usar False em casos de retomada
    "find K" : True,
    "use all variables" : False,  # Usa todas as variáveis do sistema no input search
                                  # se False, só considera os 'u' e a própria saída
    "input selection params" : {
        "max stages": 6,
        "trains per option": 1,   # inicializações para cada opção de inputs
        "search patience": 2,
        "max epochs": 1000,       # número de épocas máximo que um modelo é treinado, normalmente o early stop ativa antes
        "early stop patience": 3, # paciência do callback de early stop
        "hidden layer nodes": 8,  # fixo em um valor razoável durante todo a procura
        "horizon" : 1,            # horizonte de predição para este treinamento, recomendado: 80, 100 ou 150
        "starting y order" : 2,
        "partition size" : 1,     # para datasets muito grandes (>100k), permite pegar só uma parte para o treinamento
        "validation size": 0.3,
        "target loss": False,     # interrompe a busca quando a melhor opção de um estágio atingir esse valor
        "acceptable loss": False, # interrompe a busca se nenhuma opção melhorar o desempenho e esse valor já foi atingido
        "min delta loss": False,  # delta mínimo para ser considerado uma melhoria de desempenho
        "structure" : "DLP"       # Pode ser DLP ou LSTM, que é mais consistente
        },

    "K selection params" : {
        "K min": 8,
        "K max": 12,
        "trains per K": 1,        # recomendado: 3
        "search patience": 2,
        "max epochs": 1000,
        "early stop patience": 3,
        "horizon" : 1,            # recomendados: 20, 40, 50
        "partition size" : 1,
        "validation size": 0.3,   # recomendado: 0.3
        "test size": False,       # recomendado: 0.15, porém usar 0 se horizon = 1
        "target loss": False, 
        "min delta loss": False
        }
    }

""" Configurações de análise """

analysis_cfg = {
    "create analysis dict": True,
    "create model dict" : True,   # dicionário para exportação dos modelos ao software externo no formato combinado
    "single plots" : True,
    "multiplots" : True,
    "multiplot size" : [5, 2],
    "save plots" : True,
    "plots horizon" : "default"   # permite realizar os plots com um horizonte diferente
    }


""" Início da execução """

if exec_cfg["run"]:
    
    for y in range(exec_cfg["first y"], exec_cfg["last y"] + 1):
        
        output = 'y' + str(y)

        print("Initializing training operations for output " + output)
    
        try:
            dependency = training_dictionary[output]["dependency mask"]
        except KeyError:
            dependency = None          
            try:
                training_dictionary[output]
            except KeyError:
                training_dictionary[output] = {}
                print("No dictionary entry for " + output + ", creating a new entry")
                
        if exec_cfg["use all variables"]:
            data = raw_data
        else:
            data = du.trim_data(raw_data, output, dependency)
        
        if exec_cfg["find inputs"]:
            regressors, input_selection_results = tu.input_selection(data, output, exec_cfg["input selection params"])
                
            training_dictionary[output]["input selection results"] = input_selection_results
            training_dictionary[output]["model"] = { "regressors" : regressors }
                
            du.save_pickle(training_dictionary, "analysis\\dictionary.pickle")
        else:
            try:
                regressors = training_dictionary[output]["model"]["regressors"]
            except KeyError: 
                print("No regressors found for " + output + ", please run input selection first")
             
        if exec_cfg["find K"]:
            K, weights, K_selection_results = tu.K_selection(data, regressors, output, exec_cfg["K selection params"])
            
            training_dictionary[output]["K selection results"] = K_selection_results
            training_dictionary[output]["model"]["K"] = K
            training_dictionary[output]["model"]["weights"] = weights
            
            du.save_pickle(training_dictionary, "analysis\\dictionary.pickle")
                              
        # Esses deletes servem para deixar o Variable Explorer menos poluído
        try: del data, regressors, dependency, output, y, K, K_selection_results, weights, input_selection_results, Ny
        except: pass
        

if analysis_cfg["create analysis dict"]:
    analysis_dict = au.run_analysis(training_dictionary)
    
if analysis_cfg["create model dict"]:
    model_dict = au.create_model_dict(training_dictionary)

if analysis_cfg["single plots"]:
    loss_info = pu.single_plots(training_dictionary, raw_data, save=analysis_cfg["save plots"], horizon=analysis_cfg["plots horizon"])
    analysis_dict["plot evaluation"] = loss_info
    du.save_pickle(analysis_dict, "analysis\\analysis_dict.pickle")

if analysis_cfg["multiplots"]:
    pu.multiplots(training_dictionary, raw_data, size=analysis_cfg["multiplot size"], save=analysis_cfg["save plots"], horizon=analysis_cfg["plots horizon"])
    
# Esses deletes servem para deixar o Variable Explorer menos poluído      
del exec_cfg, analysis_cfg