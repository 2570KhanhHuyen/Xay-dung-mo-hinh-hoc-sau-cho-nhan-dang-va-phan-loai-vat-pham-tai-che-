# ♻️ Phân Loại Rác Tái Chế và Không Tái Chế bằng Deep Learning

## 🚀 Mục Tiêu Dự Án
Phát triển một mô hình học sâu giúp **phân loại rác thải thành hai loại**: **có thể tái chế** và **không thể tái chế**, nhằm hỗ trợ quá trình xử lý rác hiệu quả, giảm thiểu ô nhiễm môi trường và tối ưu hóa tái chế.

---

## 🧠 Công Nghệ Sử Dụng

- **Deep Learning** với **CNN (Convolutional Neural Network)**
- **Transfer Learning**: Mô hình **VGG19** tiền huấn luyện
- **Nền tảng**: Google Colab

---

## 📦 Thư Viện Hỗ Trợ

| Mục đích | Thư viện |
|----------|----------|
| Xử lý dữ liệu & tệp tin | `pandas`, `numpy`, `os`, `glob` |
| Học sâu & trí tuệ nhân tạo | `tensorflow`, `keras`, `VGG19`, `CNN layers` |
| Xử lý & biến đổi hình ảnh | `skimage`, `ImageDataGenerator` |
| Trực quan hóa & đánh giá | `matplotlib`, `seaborn`, `sklearn.metrics` |

---

## 📁 Mô Tả Dữ Liệu

- Tổng cộng: **2.233 hình ảnh thực tế** thu thập từ Internet.
- Chia thành:
  - **999 ảnh**: Dùng để **huấn luyện**
  - **1.234 ảnh**: Dùng để **kiểm tra và đánh giá**
- Phân loại:
  - ♻️ **Tái chế được**: chai nhựa, túi vải, hộp nhựa,...
  - 🚯 **Không tái chế được**: thức ăn thừa, bao ni lông, hộp xốp, rác hữu cơ...

---

## 🔧 Quy Trình Xây Dựng Mô Hình

### 1️⃣ Thu Thập & Chuẩn Bị Dữ Liệu
- Tải ảnh và chia thành hai nhóm: **Recycle** & **Non-Recycle**
- Chia tập train/test hợp lý để đảm bảo mô hình học tổng quát.

### 2️⃣ Tiền Xử Lý
- Resize ảnh về kích thước chuẩn `180x180`
- Chuẩn hóa pixel
- **Tăng cường dữ liệu (Data Augmentation)**: xoay, lật ảnh,...

### 3️⃣ Gán Nhãn
- Mỗi ảnh được gán nhãn tương ứng `Recycle` hoặc `Non-Recycle`

### 4️⃣ Xây Dựng Mô Hình
- Dựa trên mô hình **VGG19** đã tiền huấn luyện
- Thêm các lớp CNN, pooling và fully connected để phân loại

### 5️⃣ Huấn Luyện Mô Hình
- Huấn luyện **2 lần**:
  - Lần 1: Với dữ liệu gốc
  - Lần 2: Với dữ liệu tăng cường
- Sử dụng:
  - Loss: `categorical_crossentropy`
  - Optimizer: `Adam`
  - Callback: `ModelCheckpoint`,...

### 6️⃣ Đánh Giá Hiệu Suất
- Sử dụng **accuracy**, **confusion matrix**
- Trực quan hóa bằng **seaborn**
- Sau tăng cường dữ liệu:
  - Giảm thiên lệch
  - Cải thiện tổng quát hóa mô hình

### 7️⃣ Dự Đoán
- Load ảnh mới
- Tiền xử lý và dự đoán thuộc lớp nào
- Phân loại vẫn có thể nhầm lẫn nếu ảnh đầu vào không rõ ràng

---

## 📊 Kết Quả & Hiệu Suất

- Mô hình đạt độ chính xác cao trên tập kiểm tra
- Sau tăng cường dữ liệu:
  - Hiệu suất nhận diện **hai lớp cân bằng hơn**
  - Giảm thiểu lỗi do lệch lớp
  - Mô hình có tính **tổng quát tốt hơn**

---

## ✅ Kết Luận

Dự án đã **xây dựng thành công** một mô hình học sâu giúp phân loại **rác tái chế và không tái chế** với độ chính xác cao từ ảnh thực tế. Đây là **bước đi quan trọng** trong việc **ứng dụng AI vào lĩnh vực môi trường**, hỗ trợ công tác thu gom và xử lý rác thông minh hơn.

---

## 🔗 Liên kết
- 📒 Notebook Colab: [Google Colab](https://colab.research.google.com/drive/1TNicGlXfVTCxSl2t9IdbnM-tZ1AtYJQg)

