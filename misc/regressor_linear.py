# -*- coding: utf-8 -*-
"""
Regressão linear com rede neural
"""
import numpy as np
import tensorflow as tf

# Conjuntos gerados pela função y = 2x - 1, a qual deve ser
# identificada pela rede neural
X = np.array([1, 2, -4, 8, 0, 5, 10, 15])

Y = np.array([1, 3, -9, 15, -1, 9, 19, 29])

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1, activation='linear')
    ])

model.compile(optimizer='sgd', loss='mse')

model.fit(X, Y, epochs=200)

x_val = 7
print("resultado esperado = " + str(2*x_val - 1))
print("resultado obtido = " + str(model.predict(np.array([7]))))
