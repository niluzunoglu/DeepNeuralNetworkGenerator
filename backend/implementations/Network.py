import numpy as np
from typing import List, Callable, Optional
from .Layer import Layer


class NeuralNetwork:

    # Burada yapay sinir ağı bulunacak. 

    def __init__(self, input_dim: int, name: Optional[str] = "NeuralNetwork"):

        if input_dim <= 0:
            raise ValueError("Girdi boyutu pozitif olmalıdır.")

        self.input_dim = input_dim
        self.layers: List[Layer] = []
        self.name = name

    def add_layer(self, num_neurons: int,
                  activation_func: Callable[[np.ndarray], np.ndarray],
                  name: Optional[str] = None):
        """
        Ağa yeni bir tam bağlı (dense) katman ekler.

        Args:
            num_neurons (int): Eklenecek katmandaki nöron sayısı.
            activation_func (Callable): Katmanda kullanılacak aktivasyon fonksiyonu.
            name (Optional[str]): Katmana isteğe bağlı bir isim.
        """
        
        layer_input_dim = self.input_dim if not self.layers else self.layers[-1].num_neurons

        # Yeni katmanı oluştur
        new_layer = Layer(
            num_neurons=num_neurons,
            input_dim=layer_input_dim,
            activation_func=activation_func,
            name=name
        )

        # Bayrakları ayarla
        if not self.layers:
            new_layer.is_first_layer = True # Eklenen ilk *işlem* katmanı

        # Önceki "son katman" bayrağını temizle (varsa)
        if self.layers:
            self.layers[-1].is_output_layer = False

        # Yeni katmanı son katman olarak işaretle ve listeye ekle
        new_layer.is_output_layer = True
        self.layers.append(new_layer)

        print(f"Added layer: {new_layer}")


    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Ağ üzerinden ileri yayılımı gerçekleştirir.

        Args:
            X (np.ndarray): Ağın girdi verisi. Shape (input_dim,) olmalıdır
                             (tek örnek için).

        Returns:
            np.ndarray: Ağın son katmanının çıktısı (tahmin). Shape (output_dim,).
        """
        if not self.layers:
            raise RuntimeError("Ağa henüz hiçbir katman eklenmedi.")

        if X.shape[0] != self.input_dim:
            raise ValueError(
                f"Ağ girdisinin boyutu ({X.shape[0]}) beklenen boyutla "
                f"({self.input_dim}) eşleşmiyor."
            )

        # Katmanlar üzerinden veriyi sırayla geçir
        current_output = X
        for layer in self.layers:
            current_output = layer.forward(current_output)

        # Son katmanın çıktısını döndür
        return current_output

    def predict(self, X: np.ndarray) -> np.ndarray:
        """ İleri yayılım için bir alias (takma ad). """
        return self.forward(X)

    def summary(self):
        """ Ağın yapısını özetler. """
        print("-" * 50)
        print(f"Network Summary: {self.name}")
        print(f"Input Dimension: {self.input_dim}")
        print("-" * 50)
        if not self.layers:
            print("No layers added yet.")
        else:
            print("Layers:")
            total_params = 0
            for i, layer in enumerate(self.layers):
                print(f"  {i}: {layer}")
                # Parametre sayısını hesapla (ağırlıklar + biaslar)
                # Ağırlıklar: input_dim * num_neurons
                # Biaslar: num_neurons
                layer_params = (layer.input_dim * layer.num_neurons) + layer.num_neurons
                total_params += layer_params
                print(f"      Params: {layer_params} (W: {layer.input_dim}x{layer.num_neurons}, b: {layer.num_neurons})")
            print("-" * 50)
            print(f"Total Trainable Parameters: {total_params}")
        print("-" * 50)


# --- Örnek Kullanım ---
if __name__ == '__main__':
    # 4 özellikli bir girdi bekleyen ağ oluşturalım
    my_network = NeuralNetwork(input_dim=4)

    # Katmanları ekleyelim
    my_network.add_layer(num_neurons=5, activation_func=relu, name="Hidden1")
    my_network.add_layer(num_neurons=3, activation_func=relu, name="Hidden2")
    # Çıkış katmanı genellikle farklı bir aktivasyon kullanır (örn. sınıflandırma için sigmoid/softmax)
    my_network.add_layer(num_neurons=1, activation_func=sigmoid, name="Output")

    # Ağın özetini yazdır
    my_network.summary()

    # Rastgele bir girdi verisi oluşturalım (shape: (4,))
    input_data = np.random.rand(4)
    print(f"\nInput Data (shape {input_data.shape}):\n{input_data}")

    # İleri yayılımı çalıştırıp sonucu alalım
    prediction = my_network.predict(input_data)

    print(f"\nNetwork Prediction (shape {prediction.shape}):\n{prediction}")