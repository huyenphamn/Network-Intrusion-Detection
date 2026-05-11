from pyspark.ml.classification import GBTClassifier, LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def evaluate_model(predictions, model_name="Model"):
    """Evaluate predictions for Recall, Precision, F1, and Accuracy."""
    evaluator = MulticlassClassificationEvaluator(
        labelCol="target_label", 
        predictionCol="prediction"
    )
    
    # Calculate metrics
    recall = evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})
    precision = evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})
    f1 = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})
    accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
    
    print(f"\n--- {model_name} Performance ---")
    print(f"Recall:    {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Accuracy:  {accuracy:.4f}\n")
    
    return recall, precision, f1, accuracy

def train_baseline(train_data, test_data):
    """Train and evaluate the baseline Logistic Regression with L2."""
    lr = LogisticRegression(featuresCol="features", labelCol="target_label", maxIter=20, elasticNetParam=0.0)
    print("Training Logistic Regression...")
    model = lr.fit(train_data)
    predictions = model.transform(test_data)
    return model, evaluate_model(predictions, "Baseline Logistic Regression")

def train_gbt(train_data, test_data):
    """Train and evaluate the Gradient Boosted Tree model."""
    gbt = GBTClassifier(featuresCol="features", labelCol="target_label", maxIter=20)
    print("Training GBT...")
    model = gbt.fit(train_data)
    predictions = model.transform(test_data)
    return model, evaluate_model(predictions, "Gradient Boosted Tree")