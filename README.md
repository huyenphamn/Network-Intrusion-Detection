# Network Intrusion Detection System
An end-to-end Machine Learning pipeline built with **PySpark** and **Docker** to detect malicious network flows using the CSE-CIC-IDS-2018 dataset.

## Model Performance
After identifying and mitigating feature leakage, the GBT model achieved:
* **Recall**: 0.9982 (Prioritized to minimize false negatives in threat detection)
* **Precision**: 0.9982
* **F1-Score**: 0.9982

## Getting Started
1. **Launch the environment**:
   ```bash
   docker compose up -d
   docker compose logs ml-workspace
   ```

2. **Open the ipynb and connect to Spark kernel**
Add existing Jupyter kernel and copy "http://127.0.0.1:8888/lab?token=YOUR_TOKEN" 