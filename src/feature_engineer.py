# src/feature_engineer.py
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

class FeatureEngineer:
    def __init__(self, max_features=500):
        self.max_features = max_features
        self.tfidf_vectorizer = None
        self.count_vectorizer = None
    
    def create_tfidf_features(self, texts):
        """Create TF-IDF features from text"""
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                stop_words='english',
                ngram_range=(1, 2)
            )
            tfidf_features = self.tfidf_vectorizer.fit_transform(texts)
        else:
            tfidf_features = self.tfidf_vectorizer.transform(texts)
        
        return tfidf_features.toarray()
    
    def create_count_features(self, texts):
        """Create Count vectorizer features as alternative to Word2Vec"""
        if self.count_vectorizer is None:
            self.count_vectorizer = CountVectorizer(
                max_features=200,
                stop_words='english',
                ngram_range=(1, 1)
            )
            count_features = self.count_vectorizer.fit_transform(texts)
        else:
            count_features = self.count_vectorizer.transform(texts)
        
        return count_features.toarray()
    
    def create_metadata_features(self, df):
        """Create features from email metadata"""
        features = []
        
        for _, row in df.iterrows():
            text = str(row['text'])
            subject = str(row['subject'])
            
            # Text length features
            text_length = len(text)
            subject_length = len(subject)
            word_count = len(text.split())
            sentence_count = len(text.split('.'))
            
            # Urgency indicators
            urgency_words = ['urgent', 'asap', 'emergency', 'immediate', 'critical', 'important']
            urgency_score = sum(1 for word in urgency_words if word in text.lower())
            
            # Question indicators
            has_question = 1 if '?' in text else 0
            has_exclamation = 1 if '!' in text else 0
            
            # Action indicators
            action_words = ['review', 'approve', 'complete', 'submit', 'prepare', 'respond']
            action_score = sum(1 for word in action_words if word in text.lower())
            
            # Time indicators
            time_words = ['today', 'tomorrow', 'week', 'deadline', 'schedule', 'meeting']
            time_score = sum(1 for word in time_words if word in text.lower())
            
            # Sender domain (simplified)
            sender_domain = 1 if 'enron.com' in str(row['from']).lower() else 0
            
            features.append([
                text_length, subject_length, word_count, sentence_count,
                urgency_score, has_question, has_exclamation, 
                action_score, time_score, sender_domain
            ])
        
        return np.array(features)
    
    def create_hybrid_features(self, df):
        """Combine all feature types"""
        print("Creating hybrid features...")
        
        # TF-IDF features from full text
        print("  - Creating TF-IDF features...")
        tfidf_features = self.create_tfidf_features(df['full_text'])
        
        # Count features as alternative to Word2Vec
        print("  - Creating count features...")
        count_features = self.create_count_features(df['full_text'])
        
        # Metadata features
        print("  - Creating metadata features...")
        meta_features = self.create_metadata_features(df)
        
        # Combine all features
        hybrid_features = np.hstack([tfidf_features, count_features, meta_features])
        
        print(f"Feature shape: {hybrid_features.shape}")
        return hybrid_features

# Test code
if __name__ == "__main__":
    # Test feature engineering
    from data_loader import EnronDataLoader
    from preprocessor import TextPreprocessor
    
    # Create sample data to test
    loader = EnronDataLoader()
    X_train, X_test, y_train, y_test = loader.prepare_data()
    
    preprocessor = TextPreprocessor()
    X_train_processed = preprocessor.preprocess_dataframe(X_train)
    
    feature_engineer = FeatureEngineer()
    features = feature_engineer.create_hybrid_features(X_train_processed)
    print(f"Created features with shape: {features.shape}")