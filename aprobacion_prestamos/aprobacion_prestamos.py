import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

historial_crediticio = {0: 'malo', 1: 'regular', 2: 'bueno'}
tipo_empleo = {0: 'desempleado', 1: 'temporal', 2: 'permanente', 3: 'autónomo'}
aprobado = {0: 'no aprobado', 1: 'aprobado'}

nuevo_modelo = False
modelo = None

# Importamos los datos desde el fichero data.csv
data = pd.read_csv('data.csv')

# Separamos las características (X) y la variable objetivo (y)
X = data.drop('Decision_aprobacion', axis=1).values
y = data['Decision_aprobacion'].values

# Escalamos los datos numéricos
scaler = StandardScaler()
x_scaled = scaler.fit_transform(X)

# Dividimos los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)


def predecir_aprobacion(modelo):
    
    print("Ingrese los siguientes datos para predecir la aprobación del préstamo:")
    
    edad = int(input("Edad (18-70): "))
    ingreso_mensual = float(input("Ingreso mensual (1000-10000): "))
    deuda_actual = float(input("Deuda actual (0-20000): "))
    historial = int(input("Historial crediticio (0: malo, 1: regular, 2: bueno): "))
    anos_empleo = int(input("Años en el empleo actual (0-40): "))
    tipo_emp = int(input("Tipo de empleo (0: desempleado, 1: temporal, 2: permanente, 3: autónomo): "))
    
    # Crear un array con los datos ingresados
    datos_usuario = np.array([[edad, ingreso_mensual, deuda_actual, historial, anos_empleo, tipo_emp]])
    
    # Escalar los datos del usuario
    datos_usuario_escalados = scaler.transform(datos_usuario)
    
    # Realizar la predicción
    prediccion = modelo.predict(datos_usuario_escalados)
    
    # Interpretar la predicción
    if prediccion[0][0] >= 0.5:
        print("El préstamo ha sido aprobado.")
    else:
        print("El préstamo no ha sido aprobado.")
    
    print(f"Probabilidad de aprobación: {prediccion[0][0]:.2f}")


try:
    
    modelo = tf.keras.models.load_model('modelo_aprobacion_prestamos.h5')
    
    predecir_aprobacion(modelo)
    
except:
    
    nuevo_modelo = True
    
    modelo = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(6,)),  # Capa oculta con 16 neuronas y función de activación ReLU
        tf.keras.layers.Dense(8, activation='relu'),  # Segunda capa oculta con 8 neuronas y función de activación ReLU
        tf.keras.layers.Dense(1, activation='sigmoid')  # Capa de salida con función de activación sigmoide para clasificación binaria
    ])
    
    modelo.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
        loss = 'binary_crossentropy',
        metrics = ['accuracy',
                    tf.keras.metrics.AUC(name='auc'),
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall')]
    )
    
    historial = modelo.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2, # 20% de los datos de entrenamiento para validación
        verbose=1 # Mostrar el progreso del entrenamiento por consola
    )
    
    resultados = modelo.evaluate(X_test, y_test, verbose=1)
    print(f"Loss: {resultados[0]:.4f}, Accuracy: {resultados[1]:.4f}, AUC: {resultados[2]:.4f}, Precision: {resultados[3]:.4f}, Recall: {resultados[4]:.4f}")
    
    modelo.save('modelo_aprobacion_prestamos.h5')
    
    if nuevo_modelo:
        import matplotlib.pyplot as plt
        
        plt.plot(historial.history['loss'], label='Pérdida de entrenamiento')
        plt.title('Errores durante el entrenamiento')
        plt.xlabel('Numero de rondas de entrenamiento (epochs)')
        plt.ylabel('Pérdida (loss)')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        plt.plot(historial.history['accuracy'], label='Precisión de entrenamiento')
        plt.plot(historial.history['val_accuracy'], label='Precisión de validación')
        plt.title('Precisión durante el entrenamiento')
        plt.xlabel('Numero de rondas de entrenamiento (epochs)')
        plt.ylabel('Precisión (accuracy)')
        plt.legend()
        plt.grid(True)
        plt.show()

        