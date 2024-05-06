

class Portfolio:
    def __init__(self):
        self.cash = 50000
        self.starting_cash = 50000
        self.stocks = 0
    def cost(self, price):
        return self.cash + self.stocks * price
    def buy(self, price):
        self.cash -= price
        self.stocks += 1
    def sell(self, price):
        self.cash += price
        self.stocks -= 1
    def profit(self):
        return self.cash - self.starting_cash