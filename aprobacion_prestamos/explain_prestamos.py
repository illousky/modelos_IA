import tensorflow as tf
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt

# Cargamos el modelo
model = tf.keras.models.load_model('modelo_aprobacion_prestamos.h5')


# Función de predicción para LIME
def model_predict_np(x_numpy):
    """
        x_numpy: numpy array shape (n_samples, n_features=6)
        
        Returns: numpy array shape (n_samples,) with probabilities of approval (float values between 0 and 1)
    """
    
    y_pred = model.predict(x_numpy, verbose=0).reshape(-1)
    ypred_2 = np.vstack([1 - y_pred, y_pred]).T  # Convertir a formato (n_samples, 2) para dos clases
    return ypred_2


# Cargamos los datos de entrenamiento para LIME
data = pd.read_csv('data.csv')
X = data.drop('Decision_aprobacion', axis=1).values.astype(np.float32)
y = data['Decision_aprobacion'].values.astype(np.float32)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X = X_scaled

# Crear el explicador LIME
explainer = LimeTabularExplainer(
    training_data=X,
    feature_names=['Edad', 'Ingreso_mensual', 'Deuda_actual', 'Historial_crediticio', 'Anios_empleo', 'Tipo_empleo'],
    class_names=['No aprobado', 'Aprobado'],
    mode='classification'
)


# Elegimos una instancia para explicar (por ejemplo, la primera del conjunto de datos)
i = 2
instance = X[i]


# Generar la explicación
explanation = explainer.explain_instance(
    data_row=instance,
    predict_fn=model_predict_np,
    num_features=6 # Número de características a mostrar en la explicación
)


print("LIME: as_list():", explanation.as_list()) # pares (feature_desc, contribution)


# Guardar como HTML
explanation.save_to_file('explicacion_prestamo.html')


# Mostramos la explicación con una grafica de barras
lime_list = explanation.as_list()
feats = [f[0] for f in lime_list]
vals = [f[1] for f in lime_list]

# Obtener la probabilidad predicha por el modelo
prediction = model.predict(instance.reshape(1, -1), verbose=0)[0][0]

# Convertir a etiqueta de texto
pred_label = "Aprobado" if prediction >= 0.5 else "No aprobado"


# feats = lista de nombres de características
# vals = lista de contribuciones (pesos LIME)
# prediction = probabilidad de aprobación (float entre 0 y 1)
# pred_label = texto de resultado ("Aprobado" o "No aprobado")

# Ejemplo:
# prediction = 0.73
# pred_label = "Aprobado" if prediction >= 0.5 else "No aprobado"

feats = np.array(feats)
vals = np.array(vals)
order = np.argsort(vals)
feats = feats[order]
vals = vals[order]

# Colores: verde positivo, rojo negativo
colors = ['#2ca02c' if v > 0 else '#d62728' for v in vals]

# Color del texto del resultado
color_pred = '#2ca02c' if prediction >= 0.5 else '#d62728'

plt.figure(figsize=(12, 7))
plt.barh(feats, vals, color=colors, edgecolor='black', alpha=0.8)
plt.axvline(0, color='black', linewidth=1)

# Título principal
plt.title(
    'Contribuciones LIME para la aprobación del préstamo',
    fontsize=14,
    fontweight='bold',
    pad=35  # añade espacio para colocar el texto debajo
)

# Texto de resultado justo debajo del título
plt.text(
    0.5, 1.02,  # posición relativa (centrado horizontal, un poco debajo del título)
    f"Predicción del modelo: {pred_label} ({prediction*100:.1f}%)",
    transform=plt.gca().transAxes,
    fontsize=12,
    fontweight='bold',
    ha='center',
    color=color_pred
)

plt.xlabel('Contribución al resultado (positiva = favorece aprobación)', fontsize=12)
plt.ylabel('Características', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()
