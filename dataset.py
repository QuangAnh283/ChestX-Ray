import kagglehub

# Tải dataset Chest X-Ray (Pneumonia)
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")

print("✅ Dataset đã được tải về tại:", path)
