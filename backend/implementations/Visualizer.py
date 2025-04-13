# backend/Visualizer.py

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Dict, Optional

# NeuralNetwork sınıfını import etmemiz gerekiyor (dosya yolunu kontrol et)
# Aktivasyon fonksiyonları sadece örnek kullanım için gerekebilir.
try:
    from backend.NeuralNetwork import NeuralNetwork
    # Örnek kullanım için gerekebilir:
    # from backend.Activation import Activation
except ImportError:
    print("Warning: Could not import NeuralNetwork from backend.NeuralNetwork.")
    # Bu, dosyanın tek başına test edilmesini zorlaştırabilir
    # ama ana uygulama içinde çalışmasını engellemez.
    NeuralNetwork = None # Placeholder

class Visualizer:
    """
    Bir NeuralNetwork nesnesini katman katman görselleştirmek için bir sınıf.
    """

    def __init__(self, network: NeuralNetwork):
        """
        Görselleştiriciyi başlatır.

        Args:
            network (NeuralNetwork): Görselleştirilecek NeuralNetwork nesnesi.

        Raises:
            TypeError: Eğer girdi bir NeuralNetwork nesnesi değilse.
            ValueError: Eğer ağın görselleştirme için yeterli yapısı yoksa (örn. katman yoksa).
        """
        if NeuralNetwork is None:
             raise ImportError("NeuralNetwork class could not be imported. Cannot initialize Visualizer.")

        if not isinstance(network, NeuralNetwork):
            raise TypeError("Input must be an instance of NeuralNetwork.")
        if not network.layers:
            raise ValueError("Cannot visualize a network with no processing layers.")

        # --- Katman Boyutlarını Ağ Nesnesinden Çıkar ---
        # İlk eleman ağın girdi boyutu
        self.layer_sizes: List[int] = [network.input_dim]
        # Sonraki elemanlar her katmandaki nöron sayıları
        self.layer_sizes.extend([layer.num_neurons for layer in network.layers])
        # --- Bitti ---

        self.num_layers = len(self.layer_sizes) # Toplam katman sayısı (girdi dahil)
        self.network_name = network.name # Ağın ismini al (başlık için)

        # Çizim parametreleri
        self.neuron_radius = 0.3
        self.layer_spacing = 3.0 # Katmanlar arası yatay boşluk
        self.neuron_spacing = 0.8 # Nöronlar arası dikey boşluk

        # Nöron pozisyonlarını hesapla (burada veya draw içinde yapılabilir)
        self.neuron_positions = self._calculate_positions()


    def _calculate_positions(self) -> Dict[Tuple[int, int], Tuple[float, float]]:
        """Her nöronun (x, y) koordinatlarını hesaplar."""
        positions = {}
        if not self.layer_sizes:
            return positions

        max_neurons_in_layer = max(self.layer_sizes) if self.layer_sizes else 0
        if max_neurons_in_layer == 0:
            return positions # Boş ağ veya katman boyutu hatası

        vertical_span = (max_neurons_in_layer - 1) * self.neuron_spacing

        for layer_idx, num_neurons in enumerate(self.layer_sizes):
            x = layer_idx * self.layer_spacing
            # Nöronları dikey olarak ortalamak için başlangıç y'sini hesapla
            current_vertical_span = (num_neurons - 1) * self.neuron_spacing
            # vertical_span sıfırsa (tek nöronlu katmanlar), y_start da sıfır olmalı
            y_start = (vertical_span - current_vertical_span) / 2 if vertical_span > 0 else 0

            for neuron_idx in range(num_neurons):
                y = y_start + neuron_idx * self.neuron_spacing
                positions[(layer_idx, neuron_idx)] = (x, y)
        return positions

    def draw(self, title: Optional[str] = None):
        """Sinir ağını çizer."""
        if not self.neuron_positions:
             print("Cannot draw: No neuron positions calculated (empty or invalid network structure?).")
             return

        if title is None:
            title = f"{self.network_name} Visualization" # Ağ ismini varsayılan başlık yap

        # --- Figür ve Eksen Hazırlığı ---
        # Daha dinamik figsize hesaplaması
        max_y = max(pos[1] for pos in self.neuron_positions.values()) if self.neuron_positions else 0
        min_y = min(pos[1] for pos in self.neuron_positions.values()) if self.neuron_positions else 0
        max_x = max(pos[0] for pos in self.neuron_positions.values()) if self.neuron_positions else 0

        # Etiketler için ekstra alan bırak
        fig_height = (max_y - min_y + self.neuron_radius * 4 + self.neuron_spacing) * 0.8
        fig_width = (max_x + self.neuron_radius * 2 + self.layer_spacing) * 0.8
        # Minimum boyutlar belirle
        fig_height = max(fig_height, 4)
        fig_width = max(fig_width, 6)

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.axis('off') # Eksenleri kapat

        # --- Katman Etiketleri ---
        layer_labels = ["Input\nLayer", "Hidden Layers", "Output\nLayer"]
        label_colors = ['green', 'blue', 'red']

        # Etiket Y pozisyonu (en üst nöronun biraz üstü)
        label_y_pos = max_y + self.neuron_radius * 2.5

        # Input Label
        input_x_pos = self.neuron_positions.get((0, 0), (0,0))[0] # İlk katman x'i
        ax.text(input_x_pos, label_y_pos, layer_labels[0],
                ha='center', va='center', fontsize=14, color=label_colors[0], fontweight='bold')

        # Hidden Label(s) - Sadece 2'den fazla işlem katmanı varsa
        # self.num_layers girdi dahil toplam katman sayısıdır.
        # Yani 1 girdi + N işlem katmanı = N+1 self.num_layers
        # Gizli katmanlar 1'den başlayıp self.num_layers - 2'ye kadar gider.
        if self.num_layers > 2: # Girdi + en az 1 gizli + çıktı varsa
             hidden_start_x = self.neuron_positions.get((1, 0), (0,0))[0]
             hidden_end_x = self.neuron_positions.get((self.num_layers - 2, 0), (0,0))[0]
             hidden_center_x = hidden_start_x + (hidden_end_x - hidden_start_x) / 2
             ax.text(hidden_center_x, label_y_pos, layer_labels[1],
                     ha='center', va='center', fontsize=14, color=label_colors[1], fontweight='bold')

        # Output Label
        output_layer_idx = self.num_layers - 1
        output_x_pos = self.neuron_positions.get((output_layer_idx, 0), (0,0))[0]
        ax.text(output_x_pos, label_y_pos, layer_labels[2],
                ha='center', va='center', fontsize=14, color=label_colors[2], fontweight='bold')


        # --- Bağlantıları Çiz ---
        # self.num_layers girdi dahil toplam katman sayısıdır.
        if self.num_layers > 1:
            for layer_idx in range(self.num_layers - 1): # 0'dan başlar, sondan bir öncekine kadar
                for neuron_idx_curr in range(self.layer_sizes[layer_idx]):
                    for neuron_idx_next in range(self.layer_sizes[layer_idx + 1]):
                        pos_curr = self.neuron_positions[(layer_idx, neuron_idx_curr)]
                        pos_next = self.neuron_positions[(layer_idx + 1, neuron_idx_next)]

                        ax.plot([pos_curr[0], pos_next[0]], [pos_curr[1], pos_next[1]],
                                color='darkblue', alpha=0.6, zorder=1)


        # --- Nöronları Çiz ---
        for layer_idx in range(self.num_layers):
            for neuron_idx in range(self.layer_sizes[layer_idx]):
                x, y = self.neuron_positions[(layer_idx, neuron_idx)]

                # Katmana göre renk belirle (0: girdi, son: çıktı, arası: gizli)
                if layer_idx == 0:
                    color = 'lightgreen'
                    edge_color = 'darkgreen'
                elif layer_idx == self.num_layers - 1:
                    color = 'lightcoral'
                    edge_color = 'darkred'
                else:
                    color = 'lightblue'
                    edge_color = 'darkblue'

                circle = plt.Circle((x, y), radius=self.neuron_radius,
                                    facecolor=color, edgecolor=edge_color,
                                    linewidth=1.5, zorder=2)
                ax.add_patch(circle)

                # Nöron etiketini ekle (LaTeX formatında: a_{neuron_index}^{(layer_index)})
                label = rf'$a_{{{neuron_idx + 1}}}^{{({layer_idx})}}$'
                ax.text(x, y, label, ha='center', va='center', fontsize=10, color='black', zorder=3)

        ax.set_aspect('equal', adjustable='box')
        plt.title(title, fontsize=16, y=1.05) # Başlığı biraz yukarı al
        plt.tight_layout(pad=2.0)
        plt.show()


# --- Örnek Kullanım ---
if __name__ == '__main__':
    # Bu bloğun çalışması için NeuralNetwork ve Activation'ın import edilebilmesi lazım
    if NeuralNetwork:
        # backend.Activation'dan import varsayımıyla
        try:
             # Aktivasyon fonksiyonlarını import et
             from backend.Activation import Activation
        except ImportError:
             print("Warning: Could not import Activation class for example.")
             # Örnek kullanım için placeholder fonksiyonlar tanımlayalım
             class Activation:
                 @staticmethod
                 def relu(z): return np.maximum(0, z)
                 @staticmethod
                 def linear(z): return z
                 @staticmethod
                 def sigmoid(z): return 1 / (1 + np.exp(-z))

        print("--- Visualizer Example ---")
        # 1. Ağ Nesnesini Oluştur
        # Paylaşılan görseldeki yapı: [4, 5, 5, 5, 3]
        nn_vis_example = NeuralNetwork(input_dim=4, name="Deep Network Example")
        nn_vis_example.add_layer(num_neurons=5, activation_func=Activation.relu)
        nn_vis_example.add_layer(num_neurons=5, activation_func=Activation.relu)
        nn_vis_example.add_layer(num_neurons=5, activation_func=Activation.relu)
        nn_vis_example.add_layer(num_neurons=3, activation_func=Activation.linear) # Örnek: Lineer çıktı

        nn_vis_example.summary()

        # 2. Visualizer Nesnesini Oluştur (Ağ nesnesini vererek)
        visualizer_instance = Visualizer(network=nn_vis_example)

        # 3. Görseli Çizdir
        visualizer_instance.draw() # Varsayılan başlık ağın ismini kullanır

        print("\n--- Simple Network Example ---")
        simple_net = NeuralNetwork(input_dim=2, name="Simple MLP")
        simple_net.add_layer(num_neurons=3, activation_func=Activation.sigmoid)
        simple_net.add_layer(num_neurons=1, activation_func=Activation.sigmoid)
        simple_net.summary()
        simple_vis = Visualizer(simple_net)
        simple_vis.draw(title="Simple Network Structure")

    else:
        print("Could not run Visualizer example because NeuralNetwork class was not imported.")