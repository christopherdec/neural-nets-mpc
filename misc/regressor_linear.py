"""
Programinhasimples que treina uma rede neural de uma cada e um neurônio
utilizando a função de ativação linear.
"""

import numpy as np
import tensorflow as tf

# Foi necessário adicionar essa parte para conseguir rodar no VS Code
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus),
                  "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# Conjuntos gerados pela função y = 2x - 1, a qual deve ser
# identificada pela rede neural

X = np.array([1, 2, 5, 10, 15])

Y = np.array([1, 3, 9, 19, 29])

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1, activation='linear')
    ])

model.compile(optimizer='sgd', loss='mae')

model.fit(X, Y, epochs=500)

# Resultado esperado é 13

print(model.predict(np.array([7])))
