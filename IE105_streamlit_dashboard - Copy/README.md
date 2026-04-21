# IE105 Fraud Detection Dashboard

Dashboard Streamlit hiển thị trực quan cho 3 mô hình:
- Logistic Regression
- Isolation Forest
- Autoencoder

## Chạy local
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Cấu trúc
- `artifacts/`: model, scaler, thresholds, metrics, evaluation details
- `images/`: confusion matrix, ROC, PR, EDA
- `app.py`: mã nguồn dashboard

## Deploy Streamlit Community Cloud
1. Đẩy toàn bộ thư mục này lên GitHub.
2. Trên Streamlit Community Cloud, chọn repo và file `app.py`.
3. Deploy.
