# 🛡️ AI Accounting Assistant

### 🔒 Privacy-First • ⚡ Real-Time • 📊 Audit-Ready

The **AI Accounting Assistant** is a Streamlit app that demonstrates how **AI can protect accounting data while improving accuracy and compliance**.  

It automatically **detects and redacts PII**, flags **transaction anomalies**, and provides **instant analytics** — all with a simple file upload.

---

## 🎯 Features

- **Privacy Protection**
  - Detects SSNs, phone numbers, emails, and account numbers  
  - Automatically redacts sensitive fields  
  - No data stored or transmitted  

- **Anomaly Detection**
  - Duplicate payments  
  - Outliers (using IQR)  
  - Negative amounts  
  - Unusual patterns  

- **Instant Analytics**
  - Summary statistics (totals, averages, extremes)  
  - Data visualizations (histogram, boxplot)  
  - Interactive results  

---

## 📋 How to Use

1. Open the [Live Demo on Streamlit Cloud](https://ai-accounting-assistant-9sa7dkfi2llxvt8ng4shm7.streamlit.app/)  
2. Upload a CSV file with transactions (or use the sample data button)  
3. Review:
   - ✅ PII detection and redaction results  
   - ⚠️ Anomaly findings  
   - 📊 Summary analytics and charts  
4. Download the cleaned dataset for secure sharing  

---

## 🖼️ Example Screenshots

*(add screenshots of your app output here — redacted data table, anomaly alerts, histogram, box plot)*

---

## ⚙️ Tech Stack

- [Streamlit](https://streamlit.io/) — interactive web app  
- [pandas](https://pandas.pydata.org/) — data wrangling  
- [NumPy](https://numpy.org/) — numerical analysis  
- [matplotlib](https://matplotlib.org/) & [seaborn](https://seaborn.pydata.org/) — visualization  
- [Plotly](https://plotly.com/python/) — interactive charts  

---

👩‍💻 Author: Amy Bray (Tanner)
People Ops & HRIS Solutions | AI + Engagement | Global Workforce Programs

---

## 🛠️ Local Installation

Clone the repo:
```bash
git clone https://github.com/yourusername/ai-accounting-assistant.git
cd ai-accounting-assistant
