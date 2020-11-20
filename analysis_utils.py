import data_utils as du


def create_model_dict(dictionary):
    model_dict = {}

    for y in dictionary:
        
        try:
            regressors = dictionary[y]["model"]["regressors"]
            
            model_weights = dictionary[y]["model"]["weights"]
            
            K = dictionary[y]["model"]["K"]
            
        except KeyError:
            continue
        
        variables = list(regressors.index)

        
        weights = [model_weights[0], model_weights[2]]
        
        bias = [model_weights[1], model_weights[3]]
        
        regressors_object = {}
                
        for variable in variables:
            regressors_object[variable] = regressors[variable]
        
        model_dict[y] = {
                "weights" : weights,
                "bias" : bias,
                "n_output" : 1,
                "hidden_layer_sizes" : K,
                "regressors" : regressors_object,
                "activation" : ['tanh', 'linear']
            }
        
    du.save_pickle(model_dict, "analysis\\model_dict.pickle")
        
    return model_dict

def run_analysis(dictionary):
    
    analysis_dict = {}
    
    analysis_dict["u independent models"] = find_u_independent_models(dictionary)
            
    analysis_dict["selected regressors"] = get_selected_regressors(dictionary)
    
    analysis_dict["time info"] = get_execution_time_info(dictionary)
    
    analysis_dict["u participation"] = u_participation(dictionary)  
    
    analysis_dict["dependency masks"] = get_dependency_masks(dictionary)
    
    du.save_pickle(analysis_dict, "analysis\\analysis_dict.pickle")
    
    return analysis_dict


def get_selected_regressors(dictionary):
    selected_regressors = {}
    for y in dictionary:
        try:
            selected_regressors[y] = dictionary[y]["input selection results"]["selected regressors"]
                
        except KeyError:
            selected_regressors[y] = "None"

    return selected_regressors


def find_u_independent_models(dictionary):
    
    u_independent_outpus = []
    for y in dictionary:
        try:
            regressors = dictionary[y]["model"]["regressors"]
            
            variables = list(regressors.index)
            
            u_independent_outpus.append(y)
            
            for variable in variables:
                
                if ("u" in variable) and (regressors[variable] > 0):
                    
                    u_independent_outpus.remove(y)
                    break
                
        except KeyError:
            continue
    
    return u_independent_outpus


def get_execution_time_info(dictionary):
    info = {}
    max_K = 0
    min_K = 999
    max_inputs = 0
    min_inputs = 999
    
    K_sum = 0
    inputs_sum = 0
    num_ouputs = 0
    
    for y in dictionary:
        try:
            K_time = dictionary[y]["K selection results"]["execution time"]
            inputs_time = dictionary[y]["input selection results"]["execution time"]
            
            # Pega só a parte numérica da string e converte para float
            K_time = float(K_time.split()[0])
            inputs_time = float(inputs_time.split()[0])
            
            info[y] = "input selection: " + str(round(inputs_time, 2)) + " mins, K selection: " + \
            str(round(K_time, 2)) + " mins"
            
            K_sum += K_time
            inputs_sum += inputs_time
            num_ouputs += 1 # Não há garantia de que len(dictionary) funcionaria sempre
            
            if K_time > max_K:
                max_K = K_time
                info["Max K selection time"] = y + ": " + str( round(max_K, 2)) + " minutes"
            elif K_time < min_K:
                min_K = K_time
                info["Min K selection time"] = y + ": " + str( round(min_K, 2)) + " minutes"
                
            if inputs_time > max_inputs:
                max_inputs = inputs_time
                info["Max input selection time"] = y + ": " + str( round(max_inputs, 2)) + " minutes"
            elif inputs_time < min_inputs:
                min_inputs = inputs_time
                info["Min input selection time"] = y + ": " + str( round(min_inputs, 2)) + " minutes"
                   
        except KeyError:
            continue
    if bool(info):    
        info["Average input selection time"] = str( round(inputs_sum/num_ouputs, 2)) + " minutes"
        info["Average K selection time"] = str( round(K_sum/num_ouputs, 2)) + " minutes"
        
    return info


def u_participation(dictionary):
    
    u_participation = {}   
             
    for y in dictionary:
        try:
            regressors = dictionary[y]["model"]["regressors"]
            
            variables = regressors.index
            
            for variable in variables:
                
                if ('u' in variable) and (regressors[variable] > 0):
                    try:
                        u_participation[variable].append(y)
                    except KeyError:
                        u_participation[variable] = [y]
                  
        except KeyError:
            print("No regressors found for " + y)
            continue
    
    return u_participation


def get_dependency_masks(dictionary):
    
    masks = {}
    for y in dictionary:
        try:
            dependency = dictionary[y]["dependency mask"]
            masks[y] = dependency
            
        except KeyError:
            masks[y] = "N/A"
            
    return masks

# Antigo, em desuso
def get_K_info(dictionary):
    
    K_info = {}
    for y in dictionary:
        try:
            K_selection_results = dictionary[y]["K selection results"]
            
            for key, value in K_selection_results.items():
                
                try:
                    SSE_test = value["SSE test"]
                    K_info[y] = key + ", SSE test = " + str(SSE_test)
                except (TypeError, KeyError):
                    pass    
        except KeyError:
            pass
    
    return K_info
        