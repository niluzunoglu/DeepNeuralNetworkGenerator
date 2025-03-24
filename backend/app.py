from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

app = Flask(__name__)
CORS(app) 
model = None  
model_history = None

def create_model(layer_count, neuron_count, activation_function, input_dim):
    global model
    model = Sequential()
    model.add(Dense(neuron_count, activation=activation_function, input_dim=input_dim)) 
    for _ in range(layer_count - 1):
        model.add(Dense(neuron_count, activation=activation_function))
    model.add(Dense(1, activation='sigmoid')) 
    return model


@app.route('/api/train', methods=['POST'])
def train_model():
    data = request.get_json()
    layer_count = data['layer_count']
    neuron_count = data['neuron_count']
    activation_function = data['activation_function']
    learning_rate = data['learning_rate']
    epochs = data['epochs']

    X_train = np.random.rand(100, 5)  
    y_train = np.random.randint(0, 2, 100) 
    input_dim = X_train.shape[1] 

    global model
    model = create_model(layer_count, neuron_count, activation_function, input_dim)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    global model_history
    model_history = model.fit(X_train, y_train, epochs=epochs, verbose=0)

    loss, accuracy = model.evaluate(X_train, y_train, verbose=0)

    return jsonify({
        'loss': loss,
        'accuracy': accuracy,
        'history': model_history.history
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_data = data['input_data']
    input_data = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_data)
    return jsonify({'prediction': prediction.tolist()})  

if __name__ == '__main__':
    app.run(debug=True)