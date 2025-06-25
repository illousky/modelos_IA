import tensorflow as tf
import numpy as np

# Datos para el entrenamiento
fahrenheit = np.array([-40, -10, 0, 32, 70, 100, 212], dtype=float)
celsius = np.array([-40, -23.3, -17.8, 0, 21.1, 37.8, 100], dtype=float)


nuevo_modelo = False # Variable para indicar si se crea un nuevo modelo
modelo = None

# Si el modelo ya existe, se carga. Si no, se crea uno nuevo.
# Intentar cargar el modelo existente
try:
    modelo = tf.keras.models.load_model('modelo_fahrenheit_a_celsius.h5')
except:
    nuevo_modelo = True

    # Definición del modelo si no se carga  
    modelo = tf.keras.Sequential([
        tf.keras.Input(shape=(1,)),  # Entrada de un solo valor
        tf.keras.layers.Dense(units=1) # Capa densa con una sola neurona
    ])

    # Compilación del modelo si no se carga
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
        loss='mean_squared_error'
    )

    # Entrenamiento del modelo si no se carga
    historial = modelo.fit(fahrenheit, celsius, epochs=1000, verbose=False)

# Probar el modelo con un valor nuevo
resultado = modelo.predict(np.array([100.0]))
print("100°F son aproximadamente %.2f°C" % resultado[0][0]) # Se usa resultado[0][0] para acceder al valor predicho ya que es un np.array


# Visualización del historial de entrenamiento si el modelo es nuevo
if nuevo_modelo:
    import matplotlib.pyplot as plt

    plt.plot(historial.history['loss'])
    plt.title('Errores durante el entrenamiento')
    plt.xlabel('Numero de rondas de entrenamiento (epochs)')
    plt.ylabel('Pérdida (loss)')
    plt.grid(True)
    plt.show()


# Imprimir los pesos del modelo
capa = modelo.layers[0]
pesos = capa.get_weights()
print("Peso (multiplicador):", pesos[0][0])
print("Sesgo (bias):", pesos[1][0])

# Guardar el modelo si es nuevo
if nuevo_modelo:
    modelo.save('modelo_fahrenheit_a_celsius.h5')
