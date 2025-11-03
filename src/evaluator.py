# src/evaluator.py
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ResultsEvaluator:
    def __init__(self, models, feature_engineer, preprocessor):
        self.models = models
        self.feature_engineer = feature_engineer
        self.preprocessor = preprocessor
    
    def demo_predictions(self, email_subject, email_body, sender):
        """Demo the system with new emails"""
        # Create input dataframe
        input_data = pd.DataFrame({
            'text': [email_body],
            'subject': [email_subject],
            'from': [sender]
        })
        
        # Preprocess
        processed_data = self.preprocessor.preprocess_dataframe(input_data)
        
        # Create features
        features = self.feature_engineer.create_hybrid_features(processed_data)
        
        # Predict
        predictions, probabilities = {}, {}
        for column, model in self.models.items():
            predictions[column] = model.predict(features)[0]
            probabilities[column] = max(model.predict_proba(features)[0])
        
        # Display results
        print(f"📧 Email: {email_subject}")
        print(f"📊 Classification:")
        for column in predictions:
            print(f"  - {column}: {predictions[column]} ({probabilities[column]:.0%} confidence)")
        
        return predictions, probabilities

def run_demo():
    """Run a complete demo of the system"""
    from src.data_loader import EnronDataLoader
    from src.preprocessor import TextPreprocessor
    from src.feature_engineer import FeatureEngineer
    from src.model_trainer import MultiLabelTrainer
    
    print("🚀 Starting Email Classification Demo\n")
    
    # Load and train complete system
    loader = EnronDataLoader()
    X_train, X_test, y_train, y_test = loader.prepare_data()
    
    preprocessor = TextPreprocessor()
    X_train_processed = preprocessor.preprocess_dataframe(X_train)
    X_test_processed = preprocessor.preprocess_dataframe(X_test)
    
    feature_engineer = FeatureEngineer()
    X_train_features = feature_engineer.create_hybrid_features(X_train_processed)
    X_test_features = feature_engineer.create_hybrid_features(X_test_processed)
    
    trainer = MultiLabelTrainer()
    trainer.feature_engineer = feature_engineer
    trainer.preprocessor = preprocessor
    trainer.train_models(X_train_features, y_train)
    
    # Create evaluator
    evaluator = ResultsEvaluator(trainer.models, feature_engineer, preprocessor)
    
    # Demo predictions
    test_emails = [
        {
            'subject': "URGENT: Budget approval required",
            'body': "We need your immediate approval on the Q3 budget. Please review and respond by EOD.",
            'sender': "finance@enron.com"
        },
        {
            'subject': "Project status update", 
            'body': "Here is the weekly status report for the trading system project.",
            'sender': "project_team@enron.com"
        }
    ]
    
    print("🧪 Testing with sample emails:\n")
    for i, email in enumerate(test_emails, 1):
        print(f"Email {i}:")
        evaluator.demo_predictions(
            email['subject'],
            email['body'], 
            email['sender']
        )
        print()
    
    # Evaluate on test data
    print("📈 Final Model Performance:")
    trainer.evaluate_models(X_test_features, y_test)
    
    # Save models
    trainer.save_models()
    print("💾 Models saved successfully!")