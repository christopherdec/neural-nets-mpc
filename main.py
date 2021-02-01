"""Módulo principal

Contém a função main(), que controla o fluxo de execução do código.
É o arquivo que deve ser executado para criação e análise dos modelos.
"""
from utils import data_utils as du
from utils import training_utils as tu
from utils import analysis_utils as au
from utils import plot_utils as pu
import tensorflow as tf


def main():
    """
    Controla o fluxo de execução e a criação e análise dos modelos neurais

    Começa definindo algumas configurações do tensorflow, então importa os
    dados, os dicionários, define as configurações de execução e análise,
    e chama as rotinas do pacote de utilidades.

    Os resultados da execução ficam nos dicionários de treinamento, análise e
    de modelos, que podem ser inspecionados com o Variable Explorer do Spyder.
    """
    tf.random.set_seed(0)
    # Reseta todo estado gerado pelo tensorflow
    tf.keras.backend.clear_session()
    # Suprime mensagens de warning do tensorflow
    tf.get_logger().setLevel('ERROR')
    tf.autograph.experimental.do_not_convert

    # Carregamento de dados, no exemplo é usado o conjunto de dados levantados
    # com o modelo do simulink , que só tem uma saída 'y' e 4 variáveis de
    # entrada 'u'.
    raw_data, variable_names, Ny = du.load_data("simulink")

    # O dicionário de treinamento contém informações que vão sendo adicionadas
    # ao longo da criação dos modelos
    training_dictionary = du.load_pickle("analysis\\dictionary.pickle")

    # Para explorar os dicionários, utilizar o Variable explorer do Spyder ou
    # o variableInspector do Jupyter Lab
    # Essa inspeção é imprescindível para analisar os resultados obtidos

    """ Configurações de execução """

    # Se "create models": False, o loop de treinamentos é pulado, indo direto
    # para a análise e plots de modelos que já foram criados
    exec_cfg = {
        "create models": True,
        # first y e last y definem o intervalo de saídas para as quais serão
        # criados modelos.
        "first y": 1,
        "last y": Ny, # n�o precisa ser obrigatoriamente Ny
        "find inputs": True,  # pode-se usar False em casos de retomada
        "find K": True,  # se False, s� vai selecionar os inputs dos modelos
        "input selection params": {
            # usa todas as variáveis no input selection
            # se False, só considera as variáveis 'u' e a própria saída
            "use all variables": False,
            # inicializações para cada opção de inputs, diminui variabilidade
            "trains per option": 1,
            "max stages": 15,
            # define quantos estágios sem melhorias para interromper a busca
            "search patience": 2,
            # número máximo de épocas que um modelo é treinado, normalmente
            # o callback earlystop ativa antes de chegar em max epochs
            "max epochs": 1000,
            "early stop patience": 3,
            # fixo nessa etapa, usar um valor razoável
            "hidden layer nodes": 8,
            # horizonte de predição para esta etapa
            "horizon": 1,
            # ordem inicial das variáveis na versão decremental, ou só de y na
            # versão incremental (não mais utilizada)
            "starting order": 3,
            # para datasets muito grandes. permite utilizar só uma parte para
            # o input selection
            "partition size": 1,
            # o input selection usa o train/validation split
            "validation size": 0.3,
            # interrompe a busca quando esse valor é atingido
            "target loss": False,
            # interrompe a busca se nenhuma opção melhorar o desempenho e esse
            # valor for atingido (menos restritivo)
            "acceptable loss": False,
            # delta mínimo para considerar uma melhora como relevante
            "min delta loss": False,
            "structure": "DLP",   # DLP ou LSTM - geralmente DLP é mais rápido
            "optimizer": "adam",  # SGD (stochastic gradient descent) ou adam
            "loss": "mse",        # Função custo mean squared errors
            # Define o escalonador a ser utilizado: MinMax (só normaliza) ou
            # Standard (normaliza cada variável e divide pelo desvio padrão)
            "scaler": "Standard"
        },

        "K selection params": {
            "K min": 3,
            "K max": 10,
            "trains per K": 1,
            "search patience": 1,
            "max epochs": 1000,
            "early stop patience": 3,
            "horizon": 1,             # recomendado: 1, 20, 50, ...
            "partition size": 1,
            # K selection usa train/validation/test splits
            "validation size": 0.3,   # recomendado: 0.3
            "test size": False,       # recomendado: 0.15, 0 se horizon = 1
            "target loss": False,
            "min delta loss": False
        }
    }

    """ Configurações de análise """

    analysis_cfg = {
        "create analysis dict": True,
        # dicionário para exportação dos modelos ao software externo
        "create model dict": True,
        "single plots": True,
        "multiplots": True,
        "multiplot size": [2, 2],
        "save plots": True,
        # permite realizar os plots com um horizonte diferente do que foi
        # utilizado no K selection, padrão = "default"
        "plots horizon": 50
    }

    """ Início da execução """

    if exec_cfg["create models"]:
        training_dictionary = tu.create_models(exec_cfg,
                                               training_dictionary,
                                               raw_data)

    # O dicionário de análise contém informações sintetizadas a partir do
    # dicionário de treinamento, após sair do loop de treino
    if analysis_cfg["create analysis dict"]:
        analysis_dict = au.run_analysis(training_dictionary)
    else:
        analysis_dict = du.load_pickle("analysis\\analysis_dict.pickle")

    # O dicion�rio de modelos cont�m informa��es do modelos criados em um
    # formato combinado para exporta��o ao programa externo que o converte
    # em XML, para poder ent�o ser importado no MPA
    if analysis_cfg["create model dict"]:
        model_dict = au.create_model_dict(training_dictionary)
    else:
        model_dict = du.load_pickle("analysis\\model_dict.pickle")

    if analysis_cfg["single plots"]:
        loss_info = pu.single_plots(
            training_dictionary,
            raw_data,
            save=analysis_cfg["save plots"],
            horizon=analysis_cfg["plots horizon"])
        analysis_dict["plot evaluation"] = loss_info
        du.save_pickle(analysis_dict, "analysis\\analysis_dict.pickle")

    if analysis_cfg["multiplots"]:
        pu.multiplots(
            training_dictionary,
            raw_data,
            size=analysis_cfg["multiplot size"],
            save=analysis_cfg["save plots"],
            horizon=analysis_cfg["plots horizon"])

    return raw_data, variable_names, training_dictionary, analysis_dict, \
        model_dict


# "protective wrapper" que permite a importação deste arquivo pelo Sphinx
# para realizar a documentação sem executar o código "solto"
if __name__ == "__main__":
    raw_data, variable_names, training_dictionary, analysis_dict, \
        model_dict = main()
