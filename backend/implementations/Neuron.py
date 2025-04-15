import numpy as np
import logging
from typing import Callable, Optional 

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class Neuron:

    def __init__(self, input_dim: int, activation_func):

        self.logger = logging.getLogger(__name__)
        self.logger.debug(f"Initializing Neuron with input_dim={input_dim}")

        self.input_dim = input_dim
        self.activation_function = activation_func

        self.weights: np.ndarray = np.random.randn(input_dim) * 0.01
        self.bias: float = 0.0

        self.z: Optional[float] = None         
        self.activation: Optional[float] = None

    def forward(self, inputs: np.ndarray) -> float:
        self.logger.debug(f"Forward pass called with inputs: {inputs}")

        if inputs.shape[0] != self.input_dim:
            raise ValueError(f"Girdi boyutu ({inputs.shape[0]}) beklenen boyutla ({self.input_dim}) eşleşmiyor.")

        # 1. Lineer hesaplama: z = sum(w_i * x_i) + b = w . x + b
        # inputs shape (input_dim,), self.weights shape (input_dim,)
        self.z = np.dot(self.weights, inputs) + self.bias

        # 2. Aktivasyon fonksiyonunu uygula: a = g(z)
        self.activation = self.activation_function(self.z)

        self.logger.debug(f"Linear output (z): {self.z}, Activation output: {self.activation}")

        return self.activation

    def __str__(self):
        """Nöron hakkında bilgi veren string temsili (opsiyonel)."""
        self.logger.debug("String representation of Neuron called.")
        return (f"Neuron(Inputs: {self.input_dim}, "
                f"Weights: {self.weights.shape}, Bias: {self.bias:.4f}, "
                f"Activation: {self.activation_function.__name__})")

# --- Örnek Kullanım (Test Amaçlı) ---
if __name__ == '__main__':
    # 3 girdisi olan ve ReLU aktivasyonu kullanan bir nöron oluşturalım
    neuron1 = Neuron(input_dim=3, activation_func=relu)
    print(f"Oluşturulan Nöron 1: {neuron1}")

    # Rastgele bir girdi vektörü oluşturalım (shape: (3,))
    input_vector = np.array([0.5, -0.1, 1.2])
    print(f"Girdi Vektörü: {input_vector}")

    # İleri yayılımı çalıştıralım
    output_activation = neuron1.forward(input_vector)

    print(f"\nİleri Yayılım Sonrası:")
    print(f"  Lineer Çıktı (z): {neuron1.z:.4f}")
    print(f"  Aktivasyon (a): {neuron1.activation:.4f}")
    print(f"  Dönen Değer: {output_activation:.4f}")

    print("-" * 20)

    # 2 girdisi olan ve Sigmoid aktivasyonu kullanan bir nöron
    neuron2 = Neuron(input_dim=2, activation_func=sigmoid)
    print(f"Oluşturulan Nöron 2: {neuron2}")
    input_vector2 = np.array([-1.0, 2.0])
    print(f"Girdi Vektörü: {input_vector2}")
    output_activation2 = neuron2.forward(input_vector2)
    print(f"\nİleri Yayılım Sonrası:")
    print(f"  Lineer Çıktı (z): {neuron2.z:.4f}")
    print(f"  Aktivasyon (a): {neuron2.activation:.4f}")
    print(f"  Dönen Değer: {output_activation2:.4f}")