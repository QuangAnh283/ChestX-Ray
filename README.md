📌 Chest X-Ray Classification – Pneumonia Detection

Dự án này xây dựng một hệ thống nhận diện viêm phổi từ ảnh X-ray ngực bằng mô hình Deep Learning. Hệ thống bao gồm pipeline xử lý ảnh, huấn luyện mô hình, và API dự đoán dùng Flask.

🚀 Mục tiêu

Phân loại ảnh X-ray thành:

NORMAL

PNEUMONIA

Xây dựng mô hình CNN độ chính xác cao.

Tích hợp API dự đoán ảnh.

📂 Cấu trúc dự án
INTERNSHIP/
│
├── api/                     # API prediction
├── models/                  # Lưu model (.h5 / .pt)
│
├── dataset.py               # Load & xử lý dataset
├── train_model.py           # Train model
├── preprocess.py            # Tiền xử lý ảnh
├── predict.py               # Dự đoán ảnh X-ray
├── so_lieu.py               # Thống kê / vẽ đồ thị
│
├── .gitignore               # Bỏ dataset + file nặng
└── README (file này)

📦 Dataset

Dataset KHÔNG nằm trong repo để tránh dung lượng lớn.

Dataset Chest X-ray được tải trực tiếp (Kaggle):
🔗 Chest X-Ray Pneumonia Dataset
https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

Sau khi tải, đặt vào thư mục:

INTERNSHIP/chest_xray/

🧠 Mô hình sử dụng

Convolutional Neural Network (CNN)
ResNet18 / ResNet50
EfficientNet-B0
MobileNetV2

Bạn có thể cấu hình trong file train_model.py.

🛠️ Cách chạy dự án

1️⃣ Cài thư viện
pip install -r requirements.txt

2️⃣ Train mô hình
python train_model.py
Model sẽ được lưu vào thư mục:
models/

3️⃣ Dự đoán ảnh X-ray
python predict.py --image path/to/image.jpg

4️⃣ Chạy API
python api/app.py


API sẽ cung cấp endpoint như:

POST /predict

📊 Kết quả dự kiến

Accuracy: 85–95% tùy kiến trúc

Loss: giảm ổn định sau ~10–20 epochs

Model đạt độ tin cậy cao với ảnh chất lượng tốt.

🔍 Tiền xử lý ảnh

Resize 224×224

Chuẩn hóa pixel

Data augmentation:

Random rotation

Horizontal flip

Zoom

Brightness adjustment

🚧 Hướng phát triển

Thử nghiệm ResNet/EfficientNet

Thêm Grad-CAM để giải thích dự đoán

Triển khai UI upload ảnh

Triển khai API lên cloud (Render / Railway / HuggingFace Spaces)

👨‍💻 Tác giả

Bạch Quang Anh
Dự án được thực hiện trong kỳ thực tập — mục tiêu học tập và nghiên cứu về thị giác máy tính.
