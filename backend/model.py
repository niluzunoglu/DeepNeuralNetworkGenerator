from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def create_model(layer_count, layer_neurons, activation_function, input_dim):

    global model
    model = Sequential()
    model.add(Dense(layer_neurons[0], activation=activation_function, input_dim=input_dim))  
    for i in range(1, layer_count):
        model.add(Dense(layer_neurons[i], activation=activation_function))  
    model.add(Dense(1, activation='sigmoid'))  
    return model