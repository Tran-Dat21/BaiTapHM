import numpy as np
import pandas as pd
# Đa Biến

class MultivariateLinearRegression:
    def __init__(self, X, y):
        # X: Ma trận đặc trưng (kích thước: số mẫu m x số đặc trưng n)
        # y: Vector nhãn / giá trị thực tế (kích thước: m)
        self.input_data = np.array(X)
        self.input_label = np.array(y)
        self.m, self.n = self.input_data.shape

        # Khởi tạo vector trọng số w gồm n phần tử bằng 0 và bias b = 0
        self.w = np.zeros(self.n)
        self.b = 0.0

    def fit(self, repeat, learning_rate):
        for i in range(repeat):
            # 1. Dự đoán (Vectorization): y_predict = X . w + b
            y_predict = np.dot(self.input_data, self.w) + self.b

            # 2. Tính sai số (error)
            error = y_predict - self.input_label

            # 3. Tính đạo hàm cho toàn bộ các trọng số cùng lúc bằng ma trận chuyển vị (X.T)
            D_w = (1 / self.m) * np.dot(self.input_data.T, error)
            D_b = (1 / self.m) * np.sum(error)

            # 4. Cập nhật w và b theo Gradient Descent
            self.w -= learning_rate * D_w
            self.b -= learning_rate * D_b

            # In ra cost (MSE) mỗi 100 lần lặp để theo dõi
            if i % 10 == 0:
                cost = (1 / (2 * self.m)) * np.sum(error ** 2)
                print(f"Lần lặp {i}: Cost = {cost:.4f}")

    def predict(self, X_input):
        # Dự đoán cho dữ liệu đầu vào mới
        return np.dot(X_input, self.w) + self.b