from pyspark.sql import SparkSession

def create_spark_session(app_name="NIDS_Pipeline"):
    """Initialize and configure the Spark Session."""
    return SparkSession.builder \
        .appName(app_name) \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.driver.memory", "10g") \
        .config("spark.executor.memory", "10g") \
        .config("spark.memory.offHeap.enabled", "true") \
        .config("spark.memory.offHeap.size", "2g") \
        .config("spark.sql.codegen.wholeStage", "false") \
        .getOrCreate()

def load_csv(spark, file_path):
    """Load dataset into a PySpark DataFrame."""
    return spark.read.csv(file_path, header=True, inferSchema=True)