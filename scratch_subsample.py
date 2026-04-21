import pandas as pd
import os

data_dir = r"e:\AIO\Project\MM-ShortVideo-Rec\data\microlens-50k"
pairs_path = os.path.join(data_dir, "pairs.csv")

# Đọc toàn bộ tương tác
df = pd.read_csv(pairs_path)
print(f"Tổng số tương tác (gốc): {len(df)}")
print(f"Tổng số User (gốc): {df['user'].nunique()}")
print(f"Tổng số Item/Video (gốc): {df['item'].nunique()}")

# 1. Chọn 5,000 Item phổ biến nhất CÓ ảnh bìa
covers_dir = os.path.join(data_dir, "covers")
item_counts = df['item'].value_counts()
valid_5k_items = []

print("Đang tìm 5,000 items có chứa ảnh bìa...")
for item_id in item_counts.index:
    # Kiểm tra xem file ảnh có tồn tại thực sự không
    if os.path.exists(os.path.join(covers_dir, f"{item_id}.jpg")):
        valid_5k_items.append(item_id)
        if len(valid_5k_items) == 5000:
            break

if len(valid_5k_items) < 5000:
    print(f"Cảnh báo: Chỉ tìm thấy {len(valid_5k_items)} items có ảnh!")
else:
    print("Đã gom đủ 5,000 items có ảnh!")

df_5k = df[df['item'].isin(valid_5k_items)]

# Nếu sau khi giảm item mà có những user chỉ còn <= 1 tương tác (không đủ chia train/test loo), 
# lọc tiếp giữ lại những user có tối thiểu 2 tương tác.
user_counts = df_5k['user'].value_counts()
valid_users = user_counts[user_counts >= 2].index
df_5k = df_5k[df_5k['user'].isin(valid_users)]

print("\n--- SAU KHI LỌC ---")
print(f"Tổng số tương tác (mới): {len(df_5k)}")
print(f"Tổng số User (mới): {df_5k['user'].nunique()}")
print(f"Tổng số Item/Video (mới): {df_5k['item'].nunique()}")

# Lưu ra thư mục dataset mới "microlens-5k"
new_data_dir = r"e:\AIO\Project\MM-ShortVideo-Rec\data\microlens-5k"
os.makedirs(new_data_dir, exist_ok=True)
df_5k.to_csv(os.path.join(new_data_dir, "pairs.csv"), index=False)

print(f"\nĐã xuất dataset 5k thành công tại: {new_data_dir}")
