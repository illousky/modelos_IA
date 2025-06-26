import tensorflow as tf
import numpy as np

kilos = np.array([0, 1, 10, 20, 50, 100, 500, 1000,  2000, 5000, 10000], dtype=float)
libras = np.array([0, 2.20462, 22.0462, 44.0925, 110.231, 220.462, 1102.31, 2204.62, 4409.25, 11023.1, 22046.2], dtype=float)

nuevo_modelo = False
modelo_kilos_a_libras = None
modelo_libras_a_kilos = None
modelo = None

opcion = None

while opcion not in ['1', '2']:
    print("Qué conversión deseas realizar?")
    print("1. Kilos a Libras")
    print("2. Libras a Kilos")
    opcion = input("Selecciona una opción (1 o 2): ")
    print()
    

if opcion == '1':
    
    try:
        modelo_kilos_a_libras = tf.keras.models.load_model('modelo_kilos_a_libras.h5')
        
    except:
        nuevo_modelo = True
        
        modelo_kilos_a_libras = tf.keras.Sequential([
            tf.keras.layers.Dense(units=1, input_shape=[1])
        ])
        
        modelo_kilos_a_libras.compile(
            optimizer=tf.keras.optimizers.Adam(0.1),
            loss='mean_squared_error'
        )
        
        historial = modelo_kilos_a_libras.fit(kilos, libras, epochs=2000, verbose=False)
    
    kilos_nuevo = input("Ingrese el peso en Kilos para convertir a Libras: ")
    kilos_nuevo = float(kilos_nuevo)  # Convertir la entrada a float
    resultado = modelo_kilos_a_libras.predict(np.array([kilos_nuevo]))
    print()
    print(str(kilos_nuevo) + " Kilos son aproximadamente %.2f Libras" % resultado[0][0])
    print()
    
    modelo = modelo_kilos_a_libras
    

elif opcion == '2':

    try:
        modelo_libras_a_kilos = tf.keras.models.load_model('modelo_libras_a_kilos.h5')

    except:
        nuevo_modelo = True

        modelo_libras_a_kilos = tf.keras.Sequential([
            tf.keras.layers.Dense(units=1, input_shape=[1])
        ])

        modelo_libras_a_kilos.compile(
            optimizer=tf.keras.optimizers.Adam(0.1),
            loss='mean_squared_error'
        )

        historial = modelo_libras_a_kilos.fit(libras, kilos, epochs=2000, verbose=False)

    libras_nuevo = input("Ingrese el peso en Libras para convertir a Kilos: ")
    libras_nuevo = float(libras_nuevo)  # Convertir la entrada a float
    resultado = modelo_libras_a_kilos.predict(np.array([libras_nuevo]))
    print()
    print(str(libras_nuevo) + " Libras son aproximadamente %.2f Kilos" % resultado[0][0])
    print()
    
    modelo = modelo_libras_a_kilos
    
    
# Visualización del historial de entrenamiento si el modelo es nuevo
if nuevo_modelo:
    import matplotlib.pyplot as plt

    plt.plot(historial.history['loss'])
    plt.title('Errores durante el entrenamiento')
    plt.xlabel('Numero de rondas de entrenamiento (epochs)')
    plt.ylabel('Pérdida (loss)')
    plt.grid(True)
    plt.show()

    # Guardar el modelo entrenado
    if opcion == '1':
        modelo_kilos_a_libras.save('modelo_kilos_a_libras.h5')
    elif opcion == '2':
        modelo_libras_a_kilos.save('modelo_libras_a_kilos.h5')
        

# Imprimir los pesos del modelo
capa = modelo.layers[0]
pesos = capa.get_weights()
print("Peso (multiplicador):", pesos[0][0])
print("Sesgo (bias):", pesos[1][0])

print("Fin del programa.")
        
        

