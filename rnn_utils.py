import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Lambda, Concatenate

def rnn_sets(X, Y, output, horizon, y_order, strategy="remove"):
    
    # estratégica de insercao
    if strategy == "insert":
    
        # len(x) seria igual ao número de janelas para um horizon = 1
        num_janelas = math.ceil(len(X)/horizon)
        
        # Valores a inserir para que cada janela tenha exatamente o mesmo número de exemplos
        #insercoes = horizon - len(X) % horizon
        insercoes = int(num_janelas*horizon - (len(X)/horizon)*horizon)
        
        for i in range (insercoes):
            X = X.append(X.tail(1).copy())
            Y = Y.append(Y.tail(1).copy())
     
    # estratégica de remocao
    elif strategy == "remove":
        
        num_janelas = math.floor(len(X)/horizon)
        
        remocoes = int((len(X)/horizon)*horizon - num_janelas * horizon)   
            
        X.drop(X.tail(remocoes).index, inplace=True)       
        Y.drop(X.tail(remocoes).index, inplace=True)
        
    # Para X e Y, coloca as amostras numa lista com 'num_janelas' exemplos de sequências de 'horizon' amostragens
    X = np.split(X, num_janelas)
    Y = np.split(Y, num_janelas)
    
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

# Função de custo sum of squared errors, em desuso
def sse(y_true, y_pred):
    #print(y_true.shape)
    #print(y_pred.shape)  
    # print (tf.math.reduce_prod(y_true, axis=1).shape)
    #y_true = tf.math.reduce_prod(y_true, axis=0)
    #y_pred = tf.math.reduce_prod(y_pred, axis=0)
    #y_true = tf.reshape( y_true, [tf.shape(y_true)[0] * tf.shape(y_true)[1],  1])
    #y_pred = tf.reshape( y_pred, [tf.shape(y_pred)[0] * tf.shape(y_true)[1],  1])
    #x = tf.placeholder(tf.float32, shape=[None, 3, None])
    #dim = tf.reduce_prod(tf.shape(x)[1:])
    #x2 = tf.reshape(x, [-1, dim])
    # print((tf.math.reduce_sum(tf.math.square( tf.math.subtract(y_pred, y_true) ) )).shape)
    # outputs = tf.transpose(outputs, [1, 0, 2])   
    return tf.math.reduce_sum( tf.math.square( tf.math.subtract(y_pred, y_true) ))


""" Modelo com recursividade externa """
def rnn_model(regressors, output, horizon, K=8):
    
    y_order = regressors[output]
    
    # Calcula o total de regressores desconsiderando a variável realimentada (que entra por outro input)
    num_regressors = sum(regressors) - y_order
    
    janela = Input(shape=(horizon, num_regressors), name='janela')
    
    y_prev = Input(shape=(y_order,), name='y_prev')
    
    hidden_layer = Dense(K, activation='tanh', name='hidden')
    
    output_layer = Dense(1, activation='linear', name='output')
    
    # y_prev é conservado para ser input, já y_pred é utilizado no loop    
    y_pred = y_prev
    outputs = []
    
    for t in range (horizon):
        
        # inputs = Lambda(lambda z: z[:, t, :])(janela)
        
        inputs = janela[:, t, :]
                
        # inputs = Concatenate()([inputs, y_pred])
        
        inputs = tf.concat([inputs, y_pred], axis=1)
        
        hidden_output = hidden_layer(inputs)
        
        output = output_layer(hidden_output)

        outputs.append(output)
        
        y_pred = tf.roll(y_pred, shift=1, axis=1)
        
        # y_pred = Lambda(lambda z: z[:, 1:])(y_pred)
        
        y_pred = y_pred[:, 1:]
        
        # y_pred = Concatenate()([output, y_pred])
        
        y_pred = tf.concat([output, y_pred], axis=1)
    
    outputs = tf.stack(outputs)
    
    outputs = tf.transpose(outputs, [1, 0, 2])
    
    model = Model(inputs=[janela, y_prev], outputs=outputs)

    model.compile(optimizer='adam', loss='mse')
    
    return model


""" Modelo Sequencial com horizonte = 1 """
class h1_model(tf.keras.Model):

  def __init__(self, K):
    super(h1_model, self).__init__() # PORQUE ELE MANDA H1_MODEL JUNTO AQUI?
    self.hidden_layer = tf.keras.layers.Dense(K, activation='tanh', name='hidden')
    self.output_layer = tf.keras.layers.Dense(1, activation='linear', name='output')
    
    self.compile(optimizer='adam', loss='mse')

  def call(self, inputs):
    x = self.hidden_layer(inputs)
    return self.output_layer(x)


""" Modelo com recursividade externa implementado em classe """

class rnn_model2(tf.keras.Model):
    
    def __init__(self, horizon, K):
        super().__init__()
        self.horizon = horizon
        self.hidden_layer = tf.keras.layers.Dense(K, activation='tanh', name='hidden')
        
        self.output_layer = tf.keras.layers.Dense(1, activation='linear', name='output')
        
        self.compile(optimizer='adam', loss='mse')
    
    def call(self, inputs):
            
          y_pred = inputs[1]
          
          u = inputs[0]
          
          predictions = []
        
          for t in range(0, self.horizon):
              
              x = u[:, t, :]
              #x = Lambda(lambda z : z[:, t, :])(u)
                    
              x = tf.concat([x, y_pred], axis=1)
              #x = Concatenate()([x, y_pred])
    
              x = self.hidden_layer(x)
    
              prediction = self.output_layer(x)
    
              predictions.append(prediction)
                      
              y_pred = tf.roll(y_pred, shift=1, axis=1)
                
              y_pred = y_pred[:, 1:]
              #y_pred = Lambda(lambda z : z[:, 1:])(y_pred)
                
              y_pred = tf.concat([prediction, y_pred], axis=1)
              #y_pred = Concatenate()([prediction, y_pred])
     
          # predictions.shape => (time, batch, features)
          predictions = tf.stack(predictions)
          
          # predictions.shape => (batch, time, features)
          predictions = tf.transpose(predictions, [1, 0, 2])
          
         
          
          return predictions
  
    
""" Modelo LSTM """
class LSTM_model(tf.keras.Model):
    
    def __init__(self, horizon, K):
        super().__init__()
        self.horizon = horizon
        self.units = K
        self.lstm_cell = tf.keras.layers.LSTMCell(K)   
        # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
        self.dense = tf.keras.layers.Dense(1, activation='linear') # num_features = 1 feature pois a rede é MISO
        
        self.compile(optimizer='adam', loss='mse')
        
    """
    The first method this model needs is a warmup method to initialize its internal state based on the inputs. 
    Once trained this state will capture the relevant parts of the input history. This is equivalent to the 
    single-step LSTM model from earlier
    """ 
    def warmup(self, inputs):
        
        y0_orig = inputs[1]
        
        y0 = tf.expand_dims(y0_orig, axis=1)
        
        x = inputs[0]
        
        x = x[:, 0:1, :]
        
        x = tf.concat([x, y0], axis=2)

        # inputs.shape => (batch, time, features)
        # x.shape => (batch, lstm_units)
        x, *state = self.lstm_rnn(x)
    
        # predictions.shape => (batch, features)
        prediction = self.dense(x)
        
        y_pred = tf.roll(y0_orig, shift=1, axis=1)
        
        y_pred = Lambda(lambda z: z[:, 1:])(y_pred)
        
        y_pred = Concatenate()([prediction, y_pred])
        
        return y_pred, prediction, state
  
    
    """
    With the RNN's state, and an initial prediction you can now continue iterating the model feeding the 
    predictions at each step back as the input. The simplest approach to collecting the output predictions 
    is to use a python list, and tf.stack after the loop
    """
    def call(self, inputs, training=None):
      
      # Initialize the lstm state
      y_pred, prediction, state = self.warmup(inputs)
      
      inputs = inputs[0]
    
      # Use a TensorArray to capture dynamically unrolled outputs and insert the first prediction
      predictions = [prediction]
    
      # Run the rest of the prediction steps. Começa em 1 porque o warmup já fez a primeira predição!
      for n in range(1, self.horizon): 
      
          x = Lambda(lambda z: z[:, n, :])(inputs)
        
          x = Concatenate()([x, y_pred])
        
          # Execute one lstm step.
          x, state = self.lstm_cell(x, states=state, training=training)
        
          # Convert the lstm output to a prediction.
          prediction = self.dense(x)
        
          # Add the prediction to the output
          predictions.append(prediction)
          
         
          y_pred = tf.roll(y_pred, shift=1, axis=1)
        
          y_pred = Lambda(lambda z: z[:, 1:])(y_pred)
        
          y_pred = Concatenate()([prediction, y_pred])
    
      # predictions.shape => (time, batch, features)
      predictions = tf.stack(predictions)
      
      # predictions.shape => (batch, time, features)
      predictions = tf.transpose(predictions, [1, 0, 2])
      
      return predictions
  
    
""" Modelo GRU """
class GRU_model(tf.keras.Model):
    
    def __init__(self, horizon, K):
        super().__init__()
        self.horizon = horizon
        self.units = K
        self.gru_cell = tf.keras.layers.GRUCell(K)   
        self.gru_rnn = tf.keras.layers.RNN(self.gru_cell, return_state=True)
        self.dense = tf.keras.layers.Dense(1, activation='linear')
        
        self.compile(optimizer='adam', loss='mse')
        
    def warmup(self, inputs):
        
        y0_orig = inputs[1]
        
        y0 = tf.expand_dims(y0_orig, axis=1)
        
        x = inputs[0]
        
        x = x[:, 0:1, :]
        
        x = tf.concat([x, y0], axis=2)

        # inputs.shape => (batch, time, features)
        # x.shape => (batch, gru_units)
        x, *state = self.gru_rnn(x)
    
        # predictions.shape => (batch, features)
        prediction = self.dense(x)
        
        y_pred = tf.roll(y0_orig, shift=1, axis=1)
        
        y_pred = Lambda(lambda z: z[:, 1:])(y_pred)
        
        y_pred = Concatenate()([prediction, y_pred])
        
        return y_pred, prediction, state
  
    
    def call(self, inputs, training=None):
      
      y_pred, prediction, state = self.warmup(inputs)
      
      inputs = inputs[0]
    
      predictions = [prediction]
    
      for n in range(1, self.horizon): 
      
          x = Lambda(lambda z: z[:, n, :])(inputs)
        
          x = Concatenate()([x, y_pred])
        
          x, state = self.gru_cell(x, states=state, training=training)

          prediction = self.dense(x)

          predictions.append(prediction)
          
          y_pred = tf.roll(y_pred, shift=1, axis=1)
        
          y_pred = Lambda(lambda z: z[:, 1:])(y_pred)
        
          y_pred = Concatenate()([prediction, y_pred])
    
      predictions = tf.stack(predictions)
      
      predictions = tf.transpose(predictions, [1, 0, 2])
      
      return predictions
  
    
""" Modelo RNN Standard """
class SimpleRNN_model(tf.keras.Model):
    
    def __init__(self, horizon, K):
        super().__init__()
        self.horizon = horizon
        self.units = K
        self.simple_rnn_cell = tf.keras.layers.SimpleRNNCell(K)   
        self.simple_rnn_layer = tf.keras.layers.RNN(self.simple_rnn_cell, return_state=True)
        self.dense = tf.keras.layers.Dense(1, activation='linear')
        
        self.compile(optimizer='adam', loss='mse')
        
    def warmup(self, inputs):
        
        y0_orig = inputs[1]
        
        y0 = tf.expand_dims(y0_orig, axis=1)
        
        x = inputs[0]
        
        x = x[:, 0:1, :]
        
        x = tf.concat([x, y0], axis=2)

        # inputs.shape => (batch, time, features)
        # x.shape => (batch, gru_units)
        x, *state = self.simple_rnn_layer(x)
    
        # predictions.shape => (batch, features)
        prediction = self.dense(x)
        
        y_pred = tf.roll(y0_orig, shift=1, axis=1)
        
        y_pred = Lambda(lambda z: z[:, 1:])(y_pred)
        
        y_pred = Concatenate()([prediction, y_pred])
        
        return y_pred, prediction, state
  
    
    def call(self, inputs, training=None):
      
      y_pred, prediction, state = self.warmup(inputs)
      
      inputs = inputs[0]
    
      predictions = [prediction]
    
      for n in range(1, self.horizon): 
      
          x = Lambda(lambda z: z[:, n, :])(inputs)
        
          x = Concatenate()([x, y_pred])
        
          x, state = self.simple_rnn_cell(x, states=state, training=training)

          prediction = self.dense(x)

          predictions.append(prediction)
          
          y_pred = tf.roll(y_pred, shift=1, axis=1)
        
          y_pred = Lambda(lambda z: z[:, 1:])(y_pred)
        
          y_pred = Concatenate()([prediction, y_pred])
    
      predictions = tf.stack(predictions)
      
      predictions = tf.transpose(predictions, [1, 0, 2])
      
      return predictions




    