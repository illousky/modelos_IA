import tensorflow as tf
import numpy as np

fibonacci = np.array([
                        1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 
                        144, 233, 377, 610, 987, 1597, 2584, 4181, 6765, 10946, 
                        17711, 28657, 46368, 75025, 121393, 196418, 317811, 514229, 832040, 1346269, 
                        2178309, 3524578, 5702887, 9227465, 14930352, 24157817, 39088169, 63245986, 102334155, 165580141, 
                        267914296, 433494437, 701408733, 1134903170, 1836311903, 2971215073, 4807526976, 7778742049, 12586269025, 20365011074
                    ], dtype=float)

next_fibonacci = np.array([
                            2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 
                            233, 377, 610, 987, 1597, 2584, 4181, 6765, 10946, 17711, 
                            28657, 46368, 75025, 121393, 196418, 317811, 514229, 832040, 1346269, 2178309, 
                            3524578, 5702887, 9227465, 14930352, 24157817, 39088169, 63245986, 102334155, 165580141, 267914296, 
                            433494437, 701408733, 1134903170, 1836311903, 2971215073, 4807526976, 7778742049, 12586269025, 20365011074, 32951280099
                        ], dtype=float)

new_model = False  # Variable to indicate if a new model is created
model = None

try:
    model = tf.keras.models.load_model('modelo_fibonacci.h5')
    
except:
    new_model = True

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(1,)),  # Entrada de un solo valor
        tf.keras.layers.Dense(units=200, activation='relu'),  # Capa oculta con 10 neuronas y activación ReLU
        tf.keras.layers.Dense(units=200, activation='relu'),  # Capa oculta con 10 neuronas y activación ReLU
        tf.keras.layers.Dense(units=200, activation='relu'),  # Capa oculta con 10 neuronas y activación ReLU
        tf.keras.layers.Dense(units=200, activation='relu'),  # Capa oculta con 10 neuronas y activación ReLU
        tf.keras.layers.Dense(units=200, activation='relu'),  # Capa oculta con 10 neuronas y activación ReLU
        tf.keras.layers.Dense(units=200, activation='relu'),  # Capa oculta con 10 neuronas y activación ReLU
        tf.keras.layers.Dense(units=200, activation='relu'),  # Capa oculta con 10 neuronas y activación ReLU
        tf.keras.layers.Dense(units=200, activation='relu'),  # Capa oculta con 10 neuronas y activación ReLU
        tf.keras.layers.Dense(units=200, activation='relu'),  # Capa oculta con 10 neuronas y activación ReLU
        tf.keras.layers.Dense(units=200, activation='relu'),  # Capa oculta con 10 neuronas y activación ReLU
        tf.keras.layers.Dense(units=200, activation='relu'),  # Capa oculta con 10 neuronas y activación ReLU
        tf.keras.layers.Dense(units=1)  # Capa de salida con una sola neurona
    ])

    # Model compilation
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='mean_squared_error'
    )

    # Model training
    history = model.fit(fibonacci, next_fibonacci, epochs=25000, verbose=False)
    
    import matplotlib.pyplot as plt
    plt.plot(history.history['loss'])
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
    print("Model created and trained successfully.")
    print()

# Provide a new Fibonacci number to predict the next one
entrada = float(input("Type a Fibonacci number to predict the next one: "))
resultado = model.predict(np.array([[entrada]]))
print(f"\nThe next Fibonacci number after {entrada} is approximately {resultado[0][0]:.2f}\n")


# Save the model if a new one was created
if new_model:
    model.save('modelo_fibonacci.h5')


