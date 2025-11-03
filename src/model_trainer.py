# src/model_trainer.py
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import joblib

class MultiLabelTrainer:
    def __init__(self):
        self.models = {}
        self.feature_engineer = None
        self.preprocessor = None
    
    def train_models(self, X_features, y_labels):
        """Train separate model for each label column"""
        print("Training multi-label models...")
        
        for column in y_labels.columns:
            print(f"Training model for: {column}")
            
            # OPTIMIZED Random Forest parameters
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced',
                max_depth=8,           # ← ADD: Prevent overfitting
                min_samples_split=5,   # ← ADD: Better for small datasets  
                min_samples_leaf=2,    # ← ADD: Reduce overfitting
                max_features='sqrt'    # ← ADD: Better generalization
            )
            
            model.fit(X_features, y_labels[column])
            self.models[column] = model
            # Cross-validation score
            cv_scores = cross_val_score(model, X_features, y_labels[column], cv=3, scoring='accuracy')
            print(f"  CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        return self.models
    
    def predict_all(self, X_features):
        """Predict all labels for given features"""
        predictions = {}
        probabilities = {}
        
        for column, model in self.models.items():
            predictions[column] = model.predict(X_features)
            # Get maximum probability for each prediction
            proba = model.predict_proba(X_features)
            probabilities[column] = np.max(proba, axis=1)
        
        return predictions, probabilities
    
    def evaluate_models(self, X_test_features, y_test):
        """Evaluate model performance"""
        print("\n=== Model Evaluation ===")
        
        predictions, probabilities = self.predict_all(X_test_features)
        
        overall_accuracy = []
        for column in y_test.columns:
            print(f"\n--- {column.upper()} ---")
            accuracy = accuracy_score(y_test[column], predictions[column])
            overall_accuracy.append(accuracy)
            print(f"Accuracy: {accuracy:.3f}")
            print(classification_report(y_test[column], predictions[column]))
        
        print(f"\n=== Overall Average Accuracy: {np.mean(overall_accuracy):.3f} ===")
        return predictions
    
    def save_models(self, filepath='models/triage_models.joblib'):
        """Save all trained models"""
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'models': self.models,
            'feature_engineer': self.feature_engineer,
            'preprocessor': self.preprocessor
        }
        joblib.dump(model_data, filepath)
        print(f"Models saved to {filepath}")
    
    @classmethod
    def load_models(cls, filepath='models/triage_models.joblib'):
        """Load trained models"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No saved models found at {filepath}")
        
        model_data = joblib.load(filepath)
        trainer = cls()
        trainer.models = model_data['models']
        trainer.feature_engineer = model_data['feature_engineer']
        trainer.preprocessor = model_data['preprocessor']
        print(f"Models loaded from {filepath}")
        return trainer

# Test code
if __name__ == "__main__":
    # Test the training pipeline
    from data_loader import EnronDataLoader
    from preprocessor import TextPreprocessor
    from feature_engineer import FeatureEngineer
    
    # Load and preprocess data
    loader = EnronDataLoader()
    X_train, X_test, y_train, y_test = loader.prepare_data()
    
    preprocessor = TextPreprocessor()
    X_train_processed = preprocessor.preprocess_dataframe(X_train)
    X_test_processed = preprocessor.preprocess_dataframe(X_test)
    
    # Create features
    feature_engineer = FeatureEngineer()
    X_train_features = feature_engineer.create_hybrid_features(X_train_processed)
    X_test_features = feature_engineer.create_hybrid_features(X_test_processed)
    
    # Train models
    trainer = MultiLabelTrainer()
    trainer.feature_engineer = feature_engineer
    trainer.preprocessor = preprocessor
    trainer.train_models(X_train_features, y_train)
    
    # Evaluate
    predictions = trainer.evaluate_models(X_test_features, y_test)
    
    # Test saving and loading
    trainer.save_models()
    print("Model training test completed successfully!")