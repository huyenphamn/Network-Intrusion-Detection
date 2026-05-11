import pytest
from pyspark.sql import SparkSession
from src.preprocessing import clean_data

@pytest.fixture(scope="session")
def spark():
    return SparkSession.builder.master("local[1]").appName("pytest").getOrCreate()

def test_clean_data_removes_nulls(spark):
    """Test that missing values are successfully dropped."""
    # Create mock data with a null value
    mock_data = [(1.0, 2.0), (None, 3.0), (4.0, 5.0)]
    df = spark.createDataFrame(mock_data, ["Feature1", "Feature2"])
    
    cleaned_df = clean_data(df)
    
    assert cleaned_df.count() == 2