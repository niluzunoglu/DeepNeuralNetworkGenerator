# backend/Loss.py

import numpy as np
import logging
from typing import Optional # Gerekirse kullanılabilir

# Eğer bu dosya ayrı çalıştırılırsa veya başka yerden import edilirse diye
# temel loglama yapılandırmasının ayarlandığından emin olalım.
# Ana uygulama dosyanızda (örn: app.py) zaten yapıyorsanız bu tekrar gerekmeyebilir.
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class Loss:
    """
    Sinir ağları için yaygın kayıp (cost) fonksiyonlarını içeren sınıf.
    Fonksiyonlar statik metot olarak tanımlanmıştır.
    """

    def __init__(self, epsilon: float = 1e-15):
        """
        Loss sınıfını başlatır.

        Args:
            epsilon (float): Özellikle logaritmik kayıplarda sayısal kararlılığı
                             korumak için kullanılan küçük bir değer (örn. log(0) önlemek için).
        """
        self.logger = logging.getLogger(__name__)
        self.logger.debug("Loss class initialized.")
        # Epsilon'u sınıf içinde saklamak yerine doğrudan metotlara parametre olarak
        # vermek daha iyi olabilir, ancak buradaki yapıya uyalım.
        self.epsilon = epsilon

    @staticmethod
    def mean_squared_error(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Ortalama Kare Hata (Mean Squared Error - MSE) kaybını hesaplar.
        Genellikle regresyon problemleri için kullanılır.

        Formül: (1 / (2 * m)) * sum((y_pred - y_true)^2)
               (1/2 faktörü türevi basitleştirmek içindir, bazen kullanılmaz)

        Args:
            y_pred (np.ndarray): Modelin tahminleri (shape: (m,) veya (m, n_outputs)).
            y_true (np.ndarray): Gerçek değerler (shape: (m,) veya (m, n_outputs)).

        Returns:
            float: Hesaplanan ortalama kare hata değeri.
        """
        logger = logging.getLogger(__name__) # Statik metot içinde logger'ı al
        m = y_true.shape[0] # Örnek sayısı
        if m == 0:
            logger.warning("MSE: Hesaplama için sıfır örnek alındı.")
            return 0.0

        logger.debug(f"MSE calculating with y_pred shape {y_pred.shape} and y_true shape {y_true.shape}")
        if y_pred.shape != y_true.shape:
             logger.warning(f"MSE: y_pred shape {y_pred.shape} and y_true shape {y_true.shape} differ.")
             # Hata vermek yerine uyarı verip devam etmeyi deneyebilir veya hata fırlatabiliriz.
             # Şimdilik devam edelim, numpy broadcasting belki halleder ama riskli.
             # raise ValueError("y_pred ve y_true şekilleri farklı olamaz.")

        cost = np.sum(np.square(y_pred - y_true)) / (2 * m)
        cost = float(np.squeeze(cost)) # Skaler değere dönüştürmeyi garanti et
        logger.debug(f"MSE calculated: {cost}")
        return cost

    @staticmethod
    def binary_crossentropy(y_pred: np.ndarray, y_true: np.ndarray, epsilon: float = 1e-15) -> float:
        """
        İkili Çapraz Entropi (Binary Cross Entropy - BCE) kaybını hesaplar.
        Genellikle ikili sınıflandırma problemleri için (sigmoid aktivasyonlu çıktı katmanıyla) kullanılır.

        Formül: -(1 / m) * sum(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))

        Args:
            y_pred (np.ndarray): Modelin tahminleri (olasılıklar, 0 ile 1 arasında).
                                 Shape: (m,) veya (m, 1).
            y_true (np.ndarray): Gerçek etiketler (0 veya 1). Shape: (m,) veya (m, 1).
            epsilon (float): log(0) veya log(1) durumlarını önlemek için y_pred değerlerini
                             [epsilon, 1 - epsilon] aralığına kırpmak için kullanılır.

        Returns:
            float: Hesaplanan ikili çapraz entropi değeri.
        """
        logger = logging.getLogger(__name__) # Statik metot içinde logger'ı al
        m = y_true.shape[0]
        if m == 0:
            logger.warning("BCE: Hesaplama için sıfır örnek alındı.")
            return 0.0

        logger.debug(f"BCE calculating with y_pred shape {y_pred.shape}, y_true shape {y_true.shape}, epsilon {epsilon}")
        if y_pred.shape != y_true.shape:
             logger.warning(f"BCE: y_pred shape {y_pred.shape} and y_true shape {y_true.shape} differ.")
             # raise ValueError("y_pred ve y_true şekilleri farklı olamaz.")

        # Sayısal kararlılık için y_pred'i kırp
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        logger.debug(f"BCE: y_pred clipped to range [{epsilon}, {1 - epsilon}]")

        # Kaybı hesapla
        term1 = y_true * np.log(y_pred_clipped)
        term2 = (1 - y_true) * np.log(1 - y_pred_clipped)
        cost = -np.sum(term1 + term2) / m
        cost = float(np.squeeze(cost)) # Skaler değere dönüştürmeyi garanti et
        logger.debug(f"BCE calculated: {cost}")
        return cost

# --- Örnek Kullanım ---
if __name__ == '__main__':
    loss_calculator = Loss() # Sınıfı başlat (logger için)

    # --- MSE Örneği ---
    y_pred_mse = np.array([[2.1], [0.9], [4.8], [3.5]]) # Shape (4, 1)
    y_true_mse = np.array([[2.0], [1.0], [5.0], [3.0]]) # Shape (4, 1)
    print("\n--- MSE Example ---")
    print(f"Y Predicted:\n{y_pred_mse}")
    print(f"Y True:\n{y_true_mse}")
    mse_value = Loss.mean_squared_error(y_pred_mse, y_true_mse)
    print(f"Calculated MSE: {mse_value:.6f}")

    # --- BCE Örneği ---
    y_pred_bce = np.array([0.9, 0.2, 0.8, 0.4]) # Shape (4,) - Sigmoid çıktıları gibi
    y_true_bce = np.array([1, 0, 1, 1])         # Shape (4,) - Gerçek etiketler
    print("\n--- BCE Example ---")
    print(f"Y Predicted (Probabilities): {y_pred_bce}")
    print(f"Y True (Labels): {y_true_bce}")
    bce_value = Loss.binary_crossentropy(y_pred_bce, y_true_bce)
    print(f"Calculated BCE: {bce_value:.6f}")

    # BCE Kırpma Örneği
    y_pred_edge = np.array([1.0, 0.0, 0.7])
    y_true_edge = np.array([1, 0, 1])
    print("\n--- BCE Clipping Example ---")
    print(f"Y Predicted (Edge): {y_pred_edge}")
    print(f"Y True (Edge): {y_true_edge}")
    bce_value_edge = Loss.binary_crossentropy(y_pred_edge, y_true_edge)
    print(f"Calculated BCE (Edge): {bce_value_edge:.6f}") # NaN olmamalı