import numpy as np
import  pandas as pd
import matplotlib.pyplot as plt

#Đơn Biến
class RegressionLiner:
    def __init__(self, x, y):
        self.input_data = x
        self.input_label = y
        self.w = 0
        self.b = 0
        self.m = len(x)

    def fit(self, repeat, learningrate):
        for i in range(repeat):
            y_predict = self.w * self.input_data + self.b

            D_w = (-2/self.m)*sum(self.input_data * (self.input_label - y_predict))
            D_b = (-1/self.m)* sum(self.input_label - y_predict)

            self.w = self.w - D_w* learningrate
            self.b = self.b - D_b * learningrate

    def predict(self, x_input):
        y_predict = self.w * x_input + self.b
        return y_predict

rng = np.random.RandomState(1)
x = 10 * rng.rand(100)
y = 2 *x + 10 + rng. random(100)
regress  = RegressionLiner(x,y)
regress.fit(1000, 0.01)

y_predict = regress.predict(x)
plt.figure(figsize=(10, 6))
plt.scatter(x,y, color="green")
plt.plot(x , y_predict , color = 'red' , lw = 3)
plt.xlabel('x', size = 20)
plt.ylabel('y', size = 20)
plt.show()



