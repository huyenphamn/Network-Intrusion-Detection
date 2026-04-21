FROM quay.io/jupyter/pyspark-notebook:spark-3.5.3
WORKDIR /home/jovyan/work
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt