# app.py

from flask import Flask, request, jsonify, abort
import numpy as np
import logging

from implementations.Network import NeuralNetwork
from implementations.Activation import Activation

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
app.logger.setLevel(logging.DEBUG) 

# Global değişken tehlikeli olabilri güncelleyeceğim burayı.
model_instance: NeuralNetwork | None = None

# Burayı constants.py dosyasına taşıyıp oradan çekebilirim
ACTIVATION_MAP = {
    'relu': Activation.relu if Activation else None,
    'sigmoid': Activation.sigmoid if Activation else None,
    'linear': Activation.linear if Activation else None
}

def create_custom_model(input_dim, layer_neurons, activation_names, network_name="myNeural"):
    """
    input_dim girdi boyutu
    layer_neurons: katmandaki nöron listesi ama index = katman no olacak şekilde düşün (örn: [5, 3, 1])
    activation_names: her katman için aktivasyon fonksiyonu ismi veya tek bir isim
    network_name: ağ için isteğe bağlı isim
    """
    if NeuralNetwork is None or Activation is None:
        app.logger.error("NeuralNetwork veya Activation sınıfları yüklenemedi.")
        return None
    if not layer_neurons:
        app.logger.error("Katman nöron listesi boş olamaz.")
        return None

    network = NeuralNetwork(input_dim=input_dim, name=network_name)
    num_layers = len(layer_neurons)
    activation_funcs = []
    if isinstance(activation_names, str):
        func = ACTIVATION_MAP.get(activation_names.lower())
        if func is None:
            app.logger.error(f"Geçersiz veya desteklenmeyen aktivasyon fonksiyonu: {activation_names}")
            return None
        activation_funcs = [func] * num_layers

    elif isinstance(activation_names, list) and len(activation_names) == num_layers:
        for name in activation_names:
            func = ACTIVATION_MAP.get(name.lower())
            if func is None:
                app.logger.error(f"Geçersiz veya desteklenmeyen aktivasyon fonksiyonu: {name}")
                return None
            activation_funcs.append(func)
    else:
        app.logger.error("`layer_neurons` ve `activation_names` listelerinin boyutları eşleşmeli veya `activation_names` tek bir string olmalı.")
        return None

    try:
        for i in range(num_layers):
            num_n = layer_neurons[i]
            act_f = activation_funcs[i]
            layer_name = f"Layer_{i+1}_{act_f.__name__}" if act_f else f"Layer_{i+1}_unknown"
            network.add_layer(num_neurons=num_n, activation_func=act_f, name=layer_name)
        app.logger.info(f"'{network.name}' başarıyla oluşturuldu.")
        network.summary()
        return network
    except Exception as e:
        app.logger.error(f"Model oluşturulurken hata oluştu: {e}")
        return None

@app.route('/')
def index():
    return "flask çalışıyor devam et"

@app.route('/api/create_network', methods=['POST'])
def create_network_endpoint():
    """
    İstekten gelen parametrelerle özel NeuralNetwork modelini oluşturur.
    tek bir formla çalışmasını bekliyorum
    çünkü fronta bir buton koyduk oradan bütün verilerle tetikliyoruz.
    """
    global model_instance 

    data = request.get_json()
    if not data:
        app.logger.error("İstek gövdesi boş veya JSON formatında değil.")
        abort(400, description="Missing JSON body in request.")

    try:

        layer_neurons = data['layer_neurons'] # Örn: [5, 3, 1]
        activation_function = data['activation_function'] # Örn: "relu" veya ["relu", "relu", "sigmoid"]
        num_features = data.get('input_dim', 5) # Eğer istekte yoksa varsayılan 5 

        # Diğer parametreler (learning_rate, epochs) şu anda kullanılmıyor
        learning_rate = data.get('learning_rate', 0.01) 
        epochs = data.get('epochs', 10) 

        input_dim = num_features 

        app.logger.info(f"Model oluşturma isteği alındı: Neurons={layer_neurons}, Activation={activation_function}, InputDim={input_dim}")

        model_instance = create_custom_model(
            input_dim=input_dim,
            layer_neurons=layer_neurons,
            activation_names=activation_function
        )

        if model_instance is None:
            app.logger.error("Model oluşturulamadı.")
            abort(500, description="Failed to create the neural network model.") 

        loss = None
        accuracy = None
        history = {} 

        model_structure = {
            "name": model_instance.name,
            "input_dimension": model_instance.input_dim,
            "layers": [str(layer) for layer in model_instance.layers],
            "total_parameters": sum([(l.input_dim * l.num_neurons) + l.num_neurons for l in model_instance.layers])
        }

        app.logger.info("Model başarıyla oluşturuldu (eğitim yapılmadı).")

        return jsonify({
            'message': 'Model structure created successfully (Training not implemented).',
            'model_structure': model_structure,
            'parameters_received': data, 
            'status': {
                'loss': loss,
                'accuracy': accuracy,
                'history': history
            }
        })

    except KeyError as e:
        app.logger.error(f"İstekte eksik parametre: {e}")
        abort(400, description=f"Missing parameter in JSON body: {e}")
    except Exception as e:
        app.logger.error(f"İstek işlenirken genel hata: {e}", exc_info=True) 
        abort(500, description="An internal server error occurred.")


@app.route('/api/sendParameters', methods=['POST'])
def send_parameters():
    data = request.get_json()
    if not data:
        abort(400, description="Missing JSON body in request.")
    # Gelen tüm veriyi olduğu gibi geri döndür
    app.logger.debug(f"sendParameters called with data: {data}")
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)