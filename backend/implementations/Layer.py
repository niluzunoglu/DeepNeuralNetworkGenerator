# backend/Layer.py

import numpy as np
from typing import List, Callable, Optional
from Neuron import Neuron, sigmoid, relu, linear

class Layer:

    def __init__(self, num_neurons, input_dim, activation_func, name):
        """
        num_neurons Bu katmanda bulunacak nöron sayısı.
        input_dim Bu katmana gelen girdi sayısı (önceki katmanın nöron sayısı veya ilk katman için özellik sayısı).
        activation_func  Bu katmandaki tüm nöronlar için kullanılacak varsayılan aktivasyon fonksiyonu.
        name  
        """
        if num_neurons <= 0:
            raise ValueError("Nöron sayısı pozitif olmalıdır.")
        if input_dim <= 0:
            raise ValueError("Girdi boyutu pozitif olmalıdır.")

        self.num_neurons = num_neurons
        self.input_dim = input_dim
        self.activation_function = activation_func
        self.name = name if name else f"Layer_{num_neurons}neurons"

        # Katmandaki nöronları oluştur
        self.neurons: List[Neuron] = [
            Neuron(input_dim=self.input_dim, activation_func=self.activation_function)
            for _ in range(self.num_neurons)
        ]

        # Katmanın ilk veya son katman olup olmadığını belirten bayraklar
        # Bunlar NeuralNetwork sınıfı tarafından ayarlanacak.
        self.is_first_layer: bool = False # Teknik olarak ilk *işlem* katmanı
        self.is_output_layer: bool = False

        # İleri yayılım sırasında hesaplanacak değerler
        self.last_input: Optional[np.ndarray] = None # Bu katmana gelen son girdi (A_prev)
        self.layer_activation: Optional[np.ndarray] = None # Bu katmanın çıktısı (A)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Katmanın ileri yayılım hesaplamasını gerçekleştirir.
        Katmandaki her nöronun forward metodunu çağırır.

        Args:
            inputs (np.ndarray): Önceki katmandan gelen aktivasyonlar (A_prev).
                                 Shape (input_dim,) olmalıdır (tek örnek için).
                                 (Batch processing için (batch_size, input_dim) gerekir
                                  ama şimdilik tek örnek varsayalım).

        Returns:
            np.ndarray: Bu katmandaki tüm nöronların aktivasyon değerlerini içeren
                        vektör (A). Shape (num_neurons,).
        """
        if inputs.shape[0] != self.input_dim:
             raise ValueError(
                 f"'{self.name}' için girdi boyutu ({inputs.shape[0]}) "
                 f"beklenen boyutla ({self.input_dim}) eşleşmiyor."
             )

        self.last_input = inputs # Geri yayılım için saklanabilir

        # Her nöron için ileri yayılımı hesapla ve sonuçları topla
        activations = [neuron.forward(inputs) for neuron in self.neurons]

        # Sonuçları NumPy dizisine çevir
        self.layer_activation = np.array(activations)

        return self.layer_activation

    def __str__(self):
        """Katman hakkında bilgi veren string temsili."""
        layer_type = " (Output)" if self.is_output_layer else (" (First Hidden)" if self.is_first_layer else "")
        return (f"Layer(Name: {self.name}{layer_type}, Neurons: {self.num_neurons}, "
                f"Input Dim: {self.input_dim}, "
                f"Activation: {self.activation_function.__name__})")

    def get_weights(self) -> np.ndarray:
        """Katmanın ağırlık matrisini döndürür (shape: input_dim, num_neurons)."""
        # Her nöronun ağırlık vektörünü sütun olarak birleştir
        return np.array([neuron.weights for neuron in self.neurons]).T

    def get_biases(self) -> np.ndarray:
        """Katmanın bias vektörünü döndürür (shape: 1, num_neurons)."""
        # Her nöronun bias'ını birleştir
        return np.array([[neuron.bias for neuron in self.neurons]])