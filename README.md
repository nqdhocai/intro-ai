## **Dự án: Tối ưu số lượng Top-k tài liệu trong hệ thống Retrieval-Augmented Generation (RAG) bằng Reinforcement Learning**

---

### **1. Tóm tắt ý tưởng**

Trong các hệ thống **Retrieval-Augmented Generation (RAG)**, việc chọn ra **top-k tài liệu** từ danh sách các tài liệu được truy xuất có ảnh hưởng lớn đến chất lượng phản hồi cuối cùng. Các hệ thống hiện tại thường sử dụng **một giá trị k cố định (ví dụ: k=5 hoặc k=10)** cho mọi truy vấn, điều này thiếu tối ưu vì mỗi truy vấn cần lượng thông tin khác nhau.

Đề tài này đề xuất một cách tiếp cận học sâu: **sử dụng Reinforcement Learning (RL)** để học một chính sách dự đoán **giá trị k tối ưu** cho từng truy vấn cụ thể. Mục tiêu là cải thiện chất lượng đầu ra của hệ thống RAG bằng cách lựa chọn số lượng tài liệu phù hợp nhất thay vì lấy số lượng cố định.

---

### **2. Input - Output của mô hình RL**

#### **Input:**

* **Query** từ người dùng.
* **Danh sách tài liệu ứng viên** được truy xuất bởi retriever (e.g., BM25, DPR, ColBERT).
* **Đặc trưng của tài liệu**, gồm:

  * Điểm số từ retriever (e.g., BM25 score, dense similarity).
  * Embedding của query và các tài liệu.
  * Vị trí xếp hạng (ranking position).
  * Metadata khác nếu có (e.g., nguồn, độ dài).
* (Tuỳ chọn) **k trước đó** nếu có thông tin lịch sử.

#### **Output:**

* **Giá trị k tối ưu** cho truy vấn đầu vào.
* (Tuỳ chọn) Phân phối xác suất trên các giá trị k để phục vụ sampling hoặc uncertainty estimation.

---

### **3. Pipeline huấn luyện mô hình RL để chọn top-k**

#### **Bước 1: Chuẩn bị dữ liệu**

* Dữ liệu gồm các truy vấn và danh sách tài liệu **có sẵn nhãn ground-truth (doc IDs)**.

* Với mỗi truy vấn:

  * Truy xuất **top-Kₘₐₓ tài liệu** từ retriever (Kₘₐₓ = 20 chẳng hạn).
  * Gọi `G` là tập các **doc ID đúng (ground-truth)**, và `D_k` là tập top-k tài liệu được chọn.
  * Với mỗi giá trị k ∈ {1, 3, 5, ..., Kₘₐₓ}, tính **reward như sau**:

    $$
    \text{Reward}_k = \frac{|G \cap D_k|}{|G|}
    $$
  * Tìm giá trị k\* cho mỗi truy vấn sao cho reward lớn nhất → dùng làm “best k” để học.

* Tập huấn luyện thu được có dạng:

  ```
  {query, top_Kmax_docs, doc_features, best_k, reward@k, doc_ids}
  ```

#### **Bước 2: Mô hình hóa bài toán RL**

* **Agent**: Một mạng neural dự đoán giá trị k.
* **State**:

  * Embedding của truy vấn.
  * Đặc trưng của top-Kₘₐₓ tài liệu (score, thứ hạng, embedding, metadata).
* **Action**: Chọn một giá trị k ∈ {1, 3, 5, ..., Kₘₐₓ}.
* **Reward**:

  $$
  \text{Reward}_k = \frac{|G \cap D_k|}{|G|}
  $$

  Đây là tỷ lệ tài liệu đúng được bao phủ trong top-k → càng cao càng tốt.

#### **Bước 3: Huấn luyện với Reinforcement Learning**

* Sử dụng thuật toán học chính sách như:

  * **REINFORCE** (baseline).
  * **PPO** hoặc **A2C** nếu muốn huấn luyện ổn định hơn và hiệu quả hơn.
* Huấn luyện chính sách chọn k sao cho **reward kỳ vọng cao nhất**.
* Có thể dùng supervised warm-up trước bằng cách khởi tạo từ “best k” trong dữ liệu.

---

### **4. Lợi ích và triển vọng**

* **Tăng độ chính xác của hệ RAG** nhờ chọn số lượng tài liệu phù hợp theo từng truy vấn.
* **Tiết kiệm chi phí inference** bằng cách giới hạn tài liệu không cần thiết.
* **Thích nghi động**: Không cần cấu hình k thủ công cho từng miền kiến thức.
* Tương thích tốt với mọi retriever và các LLM downstream.