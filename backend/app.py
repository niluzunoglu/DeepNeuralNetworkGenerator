from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    """Ana sayfa."""
    return "Flask API is running!"

@app.route('/api/train', methods=['POST'])
def train_model():
    """Modeli eğitir ve sonuçları döndürür."""
    data = request.get_json()
    layer_count = data['layer_count']
    layer_neurons = data['layer_neurons']  
    activation_function = data['activation_function']
    learning_rate = data['learning_rate']
    epochs = data['epochs']

    X_train = np.random.rand(100, 5)  
    y_train = np.random.randint(0, 2, 100)  
    input_dim = X_train.shape[1]  

    global model
    model = create_model(layer_count, layer_neurons, activation_function, input_dim) 
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

@app.route('/api/sendParameters', methods=['POST'])
def send_parameters():
    """Model parametrelerini döndürür."""
    data = request.get_json()
    layer_count = data['layer_count']
    layer_neurons = data['layer_neurons']  
    activation_function = data['activation_function']
    learning_rate = data['learning_rate']
    epochs = data['epochs']

    return jsonify({
        'layer_count': layer_count,
        'layer_neurons': layer_neurons,
        'activation_function': activation_function,
        'learning_rate': learning_rate,
        'epochs': epochs
    })

if __name__ == '__main__':
    app.run(debug=True)