import numpy as np
from portfolio import Portfolio
import copy

class Layer:
    def __init__(self, input_size, output_size):
        if isinstance(input_size, int):
            self.W = np.random.randn(input_size, output_size)
        else:
            self.W = np.random.randn(*input_size, output_size)
        self.b = np.random.randn(output_size)
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    def forward(self, X):
        return self.sigmoid(np.dot(X, self.W) + self.b)

class Trader:
    def __init__(self, window_size, feature_num, hidden_sizes, output_size):
        self.portfolio = Portfolio()
        self.current_lifetime = 0
        self.layers = []
        self.window_size = window_size
        self.layers.append(Layer([window_size, feature_num], hidden_sizes[0]))
        for i in range(1, len(hidden_sizes)):
            self.layers.append(Layer(hidden_sizes[i-1], hidden_sizes[i]))
        self.layers.append(Layer(hidden_sizes[-1], output_size))
        
    def mutate(self, mutation_rate=0.1, mutation_strength=0.1):
        for layer in self.layers:
            # Mutate weights
            mask_w = np.random.rand(*layer.W.shape) < mutation_rate
            mutation_w = np.random.randn(*layer.W.shape) * mutation_strength
            layer.W += mask_w * mutation_w

            # Mutate biases
            mask_b = np.random.rand(*layer.b.shape) < mutation_rate
            mutation_b = np.random.randn(*layer.b.shape) * mutation_strength
            layer.b += mask_b * mutation_b
            
    def forward(self, X):
        z = X
        for layer in self.layers:
            z = layer.forward(z)
        return z
    def semen(self):
        return self.layers
    def make_decision(self, X):
        z = self.forward(X)
        decision = np.argmax(z)
        price = X['close'].iloc[-1]
        if decision == 0:
            self.portfolio.buy(price)
        elif decision == 2:
            self.portfolio.sell(price)
    def pass_minute(self, X):
        if self.current_lifetime + self.window_size >= len(X):
            return False
        data = X[self.current_lifetime:self.current_lifetime+self.window_size]
        self.make_decision(data)
        self.current_lifetime += 1
        return True
    def clown(self):
        cop = copy.deepcopy(self)
        cop.reset()
        return cop
    def reset(self):
        self.current_lifetime = 0
        self.portfolio = Portfolio()