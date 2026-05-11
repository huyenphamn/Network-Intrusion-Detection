import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType, StringType

def clean_data(df):
    """Standardize columns, remove nulls, and enforce numeric types."""
    cleaned_columns = [c.strip() for c in df.columns]
    df = df.toDF(*cleaned_columns)
    
    # Replace bad values with Null
    df = df.replace(float('inf'), None) \
           .replace(float('-inf'), None) \
           .replace('Infinity', None) \
           .replace('-Infinity', None) \
           .replace('NaN', None) \
           .replace('nan', None) \
           .replace('', None)
    
    df = df.dropna()
    
    string_cols_to_keep = ['Label', 'Timestamp']
    exprs = [
        F.col(c).cast(DoubleType()).alias(c) if c not in string_cols_to_keep else F.col(c)
        for c in df.columns
    ]
    df = df.select(*exprs)
            
    return df.dropna()

def apply_binary_labels(df):
    """Convert raw labels to a binary 1.0 (Attack) and 0.0 (Benign) target."""
    df = df.withColumn("Binary_Label", F.when(F.col("Label") == "Benign", "Benign").otherwise("Attack"))
    return df.withColumn("target_label", F.when(F.col("Binary_Label") == "Attack", 1.0).otherwise(0.0).cast(DoubleType()))

def balance_classes(df, seed=42):
    """Dynamically undersample the majority class to match the minority class."""
    benign_df = df.filter(F.col("Binary_Label") == "Benign")
    attack_df = df.filter(F.col("Binary_Label") == "Attack")
    
    benign_count = benign_df.count()
    attack_count = attack_df.count()
    
    # If Benign is the majority, undersample Benign
    if benign_count > attack_count:
        fraction = attack_count / benign_count
        sampled_benign = benign_df.sample(withReplacement=False, fraction=fraction, seed=seed)
        return sampled_benign.union(attack_df)
        
    # If Attack is the majority, undersample Attack
    elif attack_count > benign_count:
        fraction = benign_count / attack_count
        sampled_attack = attack_df.sample(withReplacement=False, fraction=fraction, seed=seed)
        return benign_df.union(sampled_attack)
        
    # If they are already perfectly balanced
    else:
        return df

def drop_leakage_and_strings(df):
    """Remove data leakage features and redundant string columns."""
    leakage_columns = [
        'Label', 'Binary_Label', 'Timestamp', 
        'Fwd Seg Size Min', 'Dst Port', 'Protocol'
    ]
    string_columns = [f.name for f in df.schema.fields if isinstance(f.dataType, StringType)]
    
    columns_to_drop = list(set(leakage_columns + string_columns))
    return df.drop(*[c for c in columns_to_drop if c in df.columns])
