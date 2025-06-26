import tensorflow as tf
import numpy as np

# Datos para el entrenamiento
fahrenheit = np.array([-40, -10, 0, 32, 70, 100, 212], dtype=float)
celsius = np.array([-40, -23.3, -17.8, 0, 21.1, 37.8, 100], dtype=float)

nuevo_modelo = False  # Variable para indicar si se crea un nuevo modelo
modelo_fahrenheit_a_celsius = None
modelo_celsius_a_fahrenheit = None
modelo = None

opcion = None

while opcion not in ['1', '2']:
    print("Quál modelo desea usar?")
    print("1. Fahrenheit a Celsius")
    print("2. Celsius a Fahrenheit")
    opcion = input("Ingrese 1 o 2: ")
    print()

if opcion == '1':
    
    try:
        modelo_fahrenheit_a_celsius = tf.keras.models.load_model('modelo_fahrenheit_a_celsius.h5')
        
    except:
        nuevo_modelo = True

        # Definición del modelo si no se carga
        modelo_fahrenheit_a_celsius = tf.keras.Sequential([
            tf.keras.Input(shape=(1,)),  # Entrada de un solo valor
            tf.keras.layers.Dense(units=1)  # Capa densa con una sola neurona
        ])

        # Compilación del modelo si no se carga
        modelo_fahrenheit_a_celsius.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
            loss='mean_squared_error'
        )

        # Entrenamiento del modelo si no se carga
        historial = modelo_fahrenheit_a_celsius.fit(fahrenheit, celsius, epochs=1000, verbose=False)
        
    # Probar el modelo con un valor nuevo
    fahrenheit_nuevo = input("Ingrese la temperatura en °F para convertir a °C: ")
    fahrenheit_nuevo = float(fahrenheit_nuevo)  # Convertir la entrada a float
    resultado = modelo_fahrenheit_a_celsius.predict(np.array([fahrenheit_nuevo]))  # Predecir el valor en °C
    print()
    print(str(fahrenheit_nuevo) + "°F son aproximadamente %.2f°C" % resultado[0][0])  # Se usa resultado[0][0] para acceder al valor predicho ya que es un np.array
    print()
    
    modelo = modelo_fahrenheit_a_celsius
    
elif opcion == '2':

    try:
        modelo_celsius_a_fahrenheit = tf.keras.models.load_model('modelo_celsius_a_fahrenheit.h5')

    except:
        nuevo_modelo = True

        # Definición del modelo si no se carga
        modelo_celsius_a_fahrenheit = tf.keras.Sequential([
            tf.keras.Input(shape=(1,)),  # Entrada de un solo valor
            tf.keras.layers.Dense(units=1)  # Capa densa con una sola neurona
        ])

        # Compilación del modelo si no se carga
        modelo_celsius_a_fahrenheit.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
            loss='mean_squared_error'
        )

        # Entrenamiento del modelo si no se carga
        historial = modelo_celsius_a_fahrenheit.fit(celsius, fahrenheit, epochs=1000, verbose=False)

    # Probar el modelo con un valor nuevo
    celsius_nuevo = input("Ingrese la temperatura en °C para convertir a °F: ")
    celsius_nuevo = float(celsius_nuevo)  # Convertir la entrada a float
    resultado = modelo_celsius_a_fahrenheit.predict(np.array([celsius_nuevo]))  # Predecir el valor en °F
    print()
    print(str(celsius_nuevo) + "°C son aproximadamente %.2f°F" % resultado[0][0])  # Se usa resultado[0][0] para acceder al valor predicho ya que es un np.array
    print()
    
    modelo = modelo_celsius_a_fahrenheit
    

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
    if opcion == '1':
        modelo_fahrenheit_a_celsius.save('modelo_fahrenheit_a_celsius.h5')
    elif opcion == '2':
        modelo_celsius_a_fahrenheit.save('modelo_celsius_a_fahrenheit.h5')
    print("Modelo guardado exitosamente.")
else:
    print("Modelo ya existe, no se guardó de nuevo.")
print("Fin del programa.")
    
    