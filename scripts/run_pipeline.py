import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import (
    create_spark_session, 
    load_csv, 
    clean_data, 
    apply_binary_labels, 
    balance_classes, 
    drop_leakage_and_strings,
    vectorize_features,
    train_baseline,
    train_gbt)

def main():
    spark = create_spark_session()
    
    # Mute the harmless DAGScheduler warnings
    spark.sparkContext.setLogLevel("ERROR")
    
    print("1. Loading All Data...")
    raw_df = load_csv(spark, "data/ids2018kaggle/*.csv")
    raw_df = raw_df.sample(withReplacement=False, fraction=0.10, seed=42) # Slice to test
    raw_df.cache()
    
    print(f"Total rows loaded: {raw_df.count():,}")
    
    print("2. Preprocessing & Balancing...")
    clean_df = clean_data(raw_df)
    labeled_df = apply_binary_labels(clean_df)
    balanced_df = balance_classes(labeled_df)
    final_df = drop_leakage_and_strings(balanced_df)
    
    print("3. Vectorizing Features...")
    vectorized_df = vectorize_features(final_df)
    
    print("4. Splitting Data...")
    train_data, test_data = vectorized_df.randomSplit([0.8, 0.2], seed=42)
    train_data.cache() # Lock to RAM
    
    print("5. Training Baseline Model...")
    lr_model, lr_recall = train_baseline(train_data, test_data)
    
    print("6. Training Advanced Model...")
    gbt_model, gbt_recall = train_gbt(train_data, test_data)
    
    spark.stop()

if __name__ == "__main__":
    main()