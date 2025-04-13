import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class Activation:

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.debug("Activation class initialized.")

    @staticmethod
    def sigmoid(z):
        logging.getLogger(__name__).debug(f"Sigmoid called with input: {z}")
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def relu(z):
        logging.getLogger(__name__).debug(f"ReLU called with input: {z}")
        return np.maximum(0, z)

    @staticmethod
    def linear(z):
        logging.getLogger(__name__).debug(f"Linear called with input: {z}")
        return z
