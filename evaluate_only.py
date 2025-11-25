"""
This script runs ONLY the evaluation step on an already trained model.
"""
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import argparse

# Import local modules
try:
    import config
    from data_preprocessing import create_datasets
    from evaluation import ClassificationModelEvaluator
except ImportError as e:
    print(f"Error: Failed to import a required module: {e}")
    print("Please ensure all .py files (config, data_preprocessing, etc.) are in the same directory.")
    exit(1)
except Exception as e:
    print(f"An error occurred: {e}")
    exit(1)

def evaluate(model_name):
    """
    Loads a saved model and evaluates it on the test set.
    """
    print(f"--- Initializing Evaluation for {model_name} ---")

    # 1. Load the saved model
    model_path = os.path.join(config.MODEL_DIR, f"{model_name}_best.keras")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please run train_model.py first.")
        return
        
    print(f"[INFO] 1/3: Loading model from {model_path}...")
    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    print("[INFO] 2/3: Setting up test dataset...")
    try:
        # We only need the test dataset
        _, _, test_ds = create_datasets(
            data_dir=config.DATA_DIR,
            image_size=config.IMAGE_SIZE,
            batch_size=config.BATCH_SIZE
        )
        if test_ds is None:
             raise ValueError("create_datasets returned None. Check data paths.")
             
    except Exception as e:
        print(f"Error creating datasets: {e}")
        return

    # 3. Evaluate the model
    print(f"[INFO] 3/3: Evaluating final model on test set...")
    
    # This now matches the updated evaluation.py
    evaluator = ClassificationModelEvaluator(
        model=model,
        test_dataset=test_ds, 
        model_name=model_name
    )
    eval_results = evaluator.evaluate_performance()
    
    print("--- Evaluation Complete ---")
    if eval_results:
        print(f"Final Test Accuracy: {eval_results['Accuracy']:.4f}")
        print(f"Final Test ROC-AUC: {eval_results['ROC-AUC']:.4f}")
    
    print(f"Reports saved to: {config.LOG_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SkeletAI Gender Classification Model")
    parser.add_argument(
        "--model",
        type=str,
        default=config.DEFAULT_MODEL,
        choices=config.MODELS_TO_COMPARE,
        help="Which model architecture to evaluate."
    )
    args = parser.parse_args()
    
    evaluate(args.model)