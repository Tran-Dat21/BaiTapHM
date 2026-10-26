import numpy as np
import pandas as pd
from module1 import MultivariateLinearRegression

# 1. Đọc dữ liệu
df = pd.read_csv("housing_data.csv")
X = df[["size", "n_of_bedrooms", "number_of_floors", "age_of_home"]].values
y = df["price"].values

# 2. Chuẩn hóa dữ liệu toàn bộ (Z-score normalization)
mu, sigma = X.mean(axis=0), X.std(axis=0)
X_n = (X - mu) / sigma

# 3. Khởi tạo và huấn luyện mô hình
model = MultivariateLinearRegression(X_n, y)
model.fit(repeat=1000, learning_rate=0.01)

# 4. In ra các trọng số thu được
print("Vector trọng số (w):", model.w)
print(f"Bias (b): {model.b:.4f}")

# 5. Dự đoán giá nhà

# size = float(input("Nhập diện tích nhà (size): "))
# bedrooms = float(input("Nhập số phòng ngủ (n_of_bedrooms): "))
# floors = float(input("Nhập số tầng (number_of_floors): "))
# age = float(input("Nhập tuổi của nhà (age_of_home): "))
#
# X_new = np.array([[size, bedrooms, floors, age]])
# X_new_n = (X_new - mu) / sigma
#
# price_pred = model.predict(X_new_n)
#
# print(f"Giá nhà dự đoán: {np.asarray(price_pred).item():,.2f}")

