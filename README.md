# Network Intrusion Detection System

An end-to-end Machine Learning pipeline built with **PySpark** and **Docker** to detect malicious network flows using the CSE-CIC-IDS-2018 dataset.
It is currently optimized to run for the 10 days crawled on [Kaggle](https://www.kaggle.com/datasets/solarmainframe/ids-intrusion-csv).

## Model Performance

After identifying and mitigating feature leakage, the GBT model achieved:

* **Recall**: 0.9982 (Prioritized to minimize false negatives in threat detection)
* **Precision**: 0.9982
* **F1-Score**: 0.9982

## Getting Started

1. **Launch the environment**:
Start the container:

   ```
   docker compose up -d --build
   ```

2. **Open the ipynb and connect to Spark kernel for eda notebooks**
Once the container is running:

   ```
   docker compose logs ml-workspace
   ```

Add existing Jupyter kernel and copy "<http://127.0.0.1:8888/lab?token=YOUR_TOKEN>"

3. **Run pipeline**
Once the container is running:

   ```
   docker exec -it network-intrusion-ml-workspace spark-submit scripts/run_pipeline.py
   ```

Go to [localhost:4040](http://localhost:4040) to see the native Spark UI for progress. 
