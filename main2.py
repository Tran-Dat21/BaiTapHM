import numpy as np
import pandas as pd
from module1 import MultivariateLinearRegression

# 1. Đọc dữ liệu phim
df = pd.read_csv("movie_data.csv")
X = df[["duration_min", "imdb_rating", "release_year", "num_votes_k"]].values
y = df["preference_score"].values

# 2. Chuẩn hóa dữ liệu toàn bộ (Z-score normalization)
mu, sigma = X.mean(axis=0), X.std(axis=0)
X_n = (X - mu) / sigma

# 3. Khởi tạo và huấn luyện mô hình Hồi quy tuyến tính đa biến
model = MultivariateLinearRegression(X_n, y)
model.fit(repeat=1000, learning_rate=0.01)

# 4. In ra các trọng số thu được
print("\n" + "="*40)
print("Kết quả huấn luyện mô hình gợi ý phim:")
print("Vector trọng số (w):", model.w)
print(f"Bias (b): {model.b:.4f}")
print("="*40 + "\n")

# 5. Dự đoán điểm gợi ý cho một bộ phim mới
duration = float(input("Nhập thời lượng phim (phút) [duration_min]: ") or 120)
imdb = float(input("Nhập điểm IMDb (1-10) [imdb_rating]: ") or 8.5)
year = float(input("Nhập năm phát hành [release_year]: ") or 2022)
votes = float(input("Nhập số lượt vote (ngàn lượt) [num_votes_k]: ") or 500)

X_new = np.array([[duration, imdb, year, votes]])
X_new_n = (X_new - mu) / sigma

score_pred = model.predict(X_new_n)
pred_value = np.asarray(score_pred).item()

print(f"\nĐiểm gợi ý dự đoán cho bộ phim: {pred_value:.2f} / 100")
if pred_value >= 75:
    print("=> Đánh giá: Phim RẤT ĐÁNG XEM! (Rất khuyến nghị gợi ý cho người dùng)")
elif pred_value >= 50:
    print("=> Đánh giá: Phim MỨC ĐỘ TRUNG BÌNH (Có thể gợi ý)")
else:
    print("=> Đánh giá: Phim KHÔNG NÊN GỢI Ý (Mức độ yêu thích thấp)")