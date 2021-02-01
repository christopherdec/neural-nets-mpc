"""Módulo para análise pós-criação dos modelos"""

from .data_utils import save_pickle


def create_model_dict(dictionary):
    """Cria o dicionário de modelos para exportação ao MPA, numa estrutura
    específica que combinamos.

    O .pickle salvo deve ser convertido em XML por um programa externo.

    Args:
        dictionary (dict): training_dictionary, contendo os modelos criados

    Returns:
        dict: dicionário contendo os parâmetros dos modelos criados
    """
    model_dict = {}

    for output in dictionary:

        try:
            regressors = dictionary[output]["model"]["regressors"]

            model_weights = dictionary[output]["model"]["weights"]

            K = dictionary[output]["model"]["K"]

        except KeyError:
            continue

        variables = list(regressors.index)

        # Colocanda no formato que foi combinando

        weights = [model_weights[0], model_weights[2]]

        bias = [model_weights[1], model_weights[3]]

        regressors_object = {}

        for variable in variables:
            regressors_object[variable] = regressors[variable]

        model_dict[output] = {
            "weights": weights,
            "bias": bias,
            "n_output": 1,
            "hidden_layer_sizes": K,
            "regressors": regressors_object,
            "activation": ['tanh', 'linear']
        }

    save_pickle(model_dict, "analysis\\model_dict.pickle")

    return model_dict


def run_analysis(dictionary):
    """Recebe o dicionário de treinamento, extrai e organiza informações úteis
    deste em tópicos, retornando-as no dicionário de análise.

    Para isso chama diversas funções simples do módulo de análise.

    Args:
        dictionary (dict): dicionário de treinamento

    Returns:
        dict: dicionário de análise
    """

    analysis_dict = {}

    analysis_dict["u independent models"] = find_u_independent_models(
        dictionary)

    analysis_dict["selected regressors"] = get_selected_regressors(dictionary)

    analysis_dict["time info"] = get_execution_time_info(dictionary)

    analysis_dict["u participation"] = u_participation(dictionary)

    analysis_dict["dependency masks"] = get_dependency_masks(dictionary)

    save_pickle(analysis_dict, "analysis\\analysis_dict.pickle")

    return analysis_dict


def get_selected_regressors(dictionary):
    """Pega os regressors escolhidos no input selection, para cada modelo
    criado, e junta todos num sub-dicionário.

    Args:
        dictionary (dict): dicionário de treinamento

    Returns:
        dict: dicionário com todos regressors escolhidos
    """
    selected_regressors = {}
    for output in dictionary:
        try:
            selected_regressors[output] = \
                dictionary[output]["input selection results"]["selected " +
                                                              "regressors"]

        except KeyError:
            selected_regressors[output] = "None"

    return selected_regressors


def find_u_independent_models(dictionary):
    """Retorna uma lista dos modelos cujo regressors não teve nenhuma variável
    de entrada ("u") considerada no input selection.

    Isso não é desejável, e pode indicar que o algoritmo está falhando em
    identificar as relações entrada-saída para os modelos listados.

    Idealmente esta lista estará vazia.

    Args:
        dictionary (dict): dicionário de treinamento

    Returns:
        list: lista com modelos "u" independentes
    """
    u_independent_outpus = []
    for output in dictionary:
        try:
            regressors = dictionary[output]["model"]["regressors"]

            variables = list(regressors.index)

            u_independent_outpus.append(output)

            for variable in variables:

                if ("u" in variable) and (regressors[variable] > 0):

                    u_independent_outpus.remove(output)
                    break

        except KeyError:
            continue

    return u_independent_outpus


def get_execution_time_info(dictionary):
    """Retorna informações sobre o tempo de execução para os input selection e
    K selection, incluindo tempos máximo, mínimo e a média.

    Args:
        dictionary (dict): dicionário de treinamento

    Returns:
        dict: sub-dicionário sobre os tempos de execução
    """
    info = {}
    max_K = 0
    min_K = 999
    max_inputs = 0
    min_inputs = 999

    K_sum = 0
    inputs_sum = 0
    num_ouputs = 0

    for output in dictionary:
        try:
            K_time = dictionary[output]["K selection results"]["execution" +
                                                               "time"]
            inputs_time = dictionary[output]["input selection results" +
                                             '']["execution time"]

            # Pega só a parte numérica da string e converte para float
            K_time = float(K_time.split()[0])
            inputs_time = float(inputs_time.split()[0])

            info[output] = "input selection: " + \
                str(round(inputs_time, 2)) + " mins, K selection: " + \
                str(round(K_time, 2)) + " mins"

            K_sum += K_time
            inputs_sum += inputs_time
            # Não há garantia de que len(dictionary) funcionaria sempre
            num_ouputs += 1

            if K_time > max_K:
                max_K = K_time
                info["Max K selection time"] = output + \
                    ": " + str(round(max_K, 2)) + " minutes"
            elif K_time < min_K:
                min_K = K_time
                info["Min K selection time"] = output + \
                    ": " + str(round(min_K, 2)) + " minutes"

            if inputs_time > max_inputs:
                max_inputs = inputs_time
                info["Max input selection time"] = output + \
                    ": " + str(round(max_inputs, 2)) + " minutes"
            elif inputs_time < min_inputs:
                min_inputs = inputs_time
                info["Min input selection time"] = output + \
                    ": " + str(round(min_inputs, 2)) + " minutes"

        except KeyError:
            continue
    if bool(info):
        info["Average input selection time"] = str(
            round(inputs_sum / num_ouputs, 2)) + " minutes"
        info["Average K selection time"] = str(
            round(K_sum / num_ouputs, 2)) + " minutes"

    return info


def u_participation(dictionary):
    """Retorna um sub-dicionário que indica, para cada variável de entrada
    "u", uma lista indicando em quais modelos que esta tem participação.

    Exemplo: "u1 = [y1, y5, y12]" indica que u1 é variável de entrada para
    os modelos de y1, y5 e y12.

    Args:
        dictionary (dict): dicionário de treinamento

    Returns:
        dict: dicionário com a partipação das variáveis de entrada
    """
    u_participation = {}

    for output in dictionary:
        try:
            regressors = dictionary[output]["model"]["regressors"]

            variables = regressors.index

            for variable in variables:

                if ('u' in variable) and (regressors[variable] > 0):
                    try:
                        u_participation[variable].append(output)
                    except KeyError:
                        u_participation[variable] = [output]

        except KeyError:
            print("No regressors found for " + output)
            continue

    return u_participation


def get_dependency_masks(dictionary):
    """Retorna um sub-dicionário com as máscaras de dependência
    informadas para o input selection de cada modelo.

    Geralmente não são informadas, e por padrão todas as variáveis de entrada
    "u" e a própria saída "y" são utilizadas no input selection de cada
    modelo. Neste caso, a string "default" é registrada como a máscara usada.

    Args:
        dictionary (dict): dicionário de treinamento

    Returns:
        dict: sub-dicionário com as máscaras de dependência
    """
    masks = {}
    for output in dictionary:
        try:
            dependency = dictionary[output]["dependency mask"]
            masks[output] = dependency

        except KeyError:
            masks[output] = "default"

    return masks
