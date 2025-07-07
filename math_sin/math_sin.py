import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

try:
    model = torch.load('math_sin_model.pth', weights_only=False)  # Load the saved model
    model.eval()  # Set the model to evaluation mode

    # If the model is loaded successfully, we can use it to make predictions
    x = input("Enter a value for x: ")
    x = float(x)
    original_x = x  # Store the original value of x for later use

    negative = False
    if x < 0:
        negative = True
    
    # Normalize x to be in range [-2π, 2π]
    x = x % (2 * np.pi)

    if negative:
        x = -x

    x_tensor = torch.tensor([[x]], dtype=torch.float32)  # Convert input to tensor

    # Make a prediction using the loaded model
    with torch.no_grad():  # No necesitamos gradientes para inferencia
        y_pred = model(x_tensor)

    print(f"Model prediction: {y_pred.item():.6f}")
    print(f"Real value of sin({original_x}): {np.sin(original_x)}")


except FileNotFoundError:

    # Entry point for the PyTorch model
    x = np.linspace(-2*np.pi, 2*np.pi, 1000).reshape(-1, 1) # 1000 samples from -2π to 2π
    y = np.sin(x) # Sine values for the samples

    # Convert numpy arrays to PyTorch tensors
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # Define the neural network model
    model = nn.Sequential(
        nn.Linear(1, 128),  # Input layer with 1 input feature and 128 hidden units
        nn.ReLU(),         # Activation function
        nn.Linear(128, 128), # Hidden layer with 128 hidden units
        nn.ReLU(),         # Activation function
        nn.Linear(128, 1)   # Output layer with 1 output feature
    )

    # Loss function and optimizer
    loss_function = nn.MSELoss()  # Mean Squared Error loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)  # Adam optimizer

    loss_history = []  # To store loss values for plotting

    # Training loop
    for epoch in range(15000):

        model.train()  # Set the model to training mode
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(x_tensor)  # Forward pass
        loss = loss_function(outputs, y_tensor)  # Compute the loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update the model parameters

        loss_history.append(loss.item())  # Store the loss value

        if epoch % 100 == 0:  # Print loss every 100 epochs
            print(f'Epoch [{epoch}/15000], Loss: {loss.item():.5f}')

    # Save the trained model
    torch.save(model, 'math_sin_model.pth')
    # The model is now trained and saved as 'math_sin_model.pth'

    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        pred = model(x_tensor).numpy()

    plt.plot(x, y, label="sin(x)")
    plt.plot(x, pred, label="Predicción NN", linestyle='dashed')
    plt.legend()
    plt.title("Aproximación de sin(x) con red neuronal")
    plt.grid(True)
    plt.show()

    # Show the loss history
    plt.figure()
    plt.plot(loss_history)
    plt.title("Evolución del error (loss) durante el entrenamiento")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.show()
