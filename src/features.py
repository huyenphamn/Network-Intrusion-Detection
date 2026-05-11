from pyspark.ml.feature import VectorAssembler

def vectorize_features(df, target_col="target_label"):
    """Assemble all numerical columns into a single feature vector."""
    feature_columns = [c for c in df.columns if c != target_col]
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    
    vectorized_df = assembler.transform(df)
    return vectorized_df.select("features", target_col)