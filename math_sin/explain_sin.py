import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class SinActivation(nn.Module):
    def forward(self, x):
        return torch.sin(x)
    
    
model = torch.load('math_sin_model.pth', map_location='cpu', weights_only=False)  # Load the saved model
model.eval()  # Set the model to evaluation mode
    

def preprocess_raw_x_array(x_raw):
    """
        x_raw: numpy array shape (n_samples,) or (n_samples, 1)
        
        Applies the preprocessing used during training:
        - Takes absolute value
        - Normalizes to range [0, 2π]
        
        Returns array of shape (n_samples, 1)
    """
    
    x1d = np.abs(x_raw).reshape(-1)  # Ensure 1D array and take absolute value
    
    processed = []
    
    for x in x1d:
        
        negative = False
        
        if x < 0:
            negative = True
            
        if negative:
            x = -x
            
        # Normalize x to be in range [0, 2π]
        x = x % (2 * np.pi)
        processed.append([x])
    
    return np.array(processed, dtype=np.float32)


def model_predict_np(x_numpy):
    """
        x_numpy: numpy array shape (n_samples, n_features=1)
        
        Returns: numpy array shape (n_samples,) with model predictions (float values)
    """ 
    
    x_processed = preprocess_raw_x_array(x_numpy) # Preprocess input
    
    x_tensor = torch.tensor(x_processed, dtype=torch.float32) # Convert input to tensor
    
    with torch.no_grad():  # No necesitamos gradientes para inferencia
        y_pred_tensor = model(x_tensor).cpu().numpy().reshape(-1)  # Get model predictions as numpy array
        
    return y_pred_tensor
            
            

# Datos de entrenamiento (usados para LIME)
X_train = np.linspace(-2*np.pi, 2*np.pi, 1000).reshape(-1,1).astype(np.float32)
y_train = np.sin(X_train)

from lime import lime_tabular

explainer_lime = lime_tabular.LimeTabularExplainer(
    training_data=X_train,
    feature_names=['x'],
    mode='regression',
    discretize_continuous=False
)

# instancia a explicar (raw input) — por ejemplo x=15
x_to_explain = np.array([15.0])  # raw single value
exp = explainer_lime.explain_instance(
    x_to_explain,     # espera 1D vector (n_features,)
    model_predict_np,
    num_samples=2000
)

print("LIME: as_list():", exp.as_list())   # pares (feature_desc, contribution)
# Dibujar un barplot simple con los componentes devueltos por LIME
lime_list = exp.as_list()
feats = [f[0] for f in lime_list]
vals  = [f[1] for f in lime_list]

plt.figure()
plt.barh(feats, vals)
plt.title("LIME contributions (x=15)")
plt.xlabel("Contribution to prediction")
plt.show()
