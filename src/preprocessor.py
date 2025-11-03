# src/preprocessor.py
import nltk
import re
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.stop_words.update(['subject', 're', 'fw', 'fwd', 'dear', 'regards', 'thanks', 'hello'])
    
    def clean_text(self, text):
        """Clean and normalize text"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase and remove special characters
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())  # Remove extra whitespace
        
        return text
    
    def tokenize(self, text):
        """Tokenize and lemmatize text"""
        tokens = word_tokenize(text)
        
        # Remove stopwords and short tokens, then lemmatize
        processed_tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        return processed_tokens
    
    def preprocess_dataframe(self, df):
        """Preprocess entire dataframe"""
        print("Preprocessing text data...")
        
        # Clean all text columns
        df_processed = df.copy()
        df_processed['text_clean'] = df_processed['text'].apply(self.clean_text)
        df_processed['subject_clean'] = df_processed['subject'].apply(self.clean_text)
        
        # Tokenize
        df_processed['text_tokens'] = df_processed['text_clean'].apply(self.tokenize)
        df_processed['subject_tokens'] = df_processed['subject_clean'].apply(self.tokenize)
        
        # Combine all text for full email content
        df_processed['full_text'] = df_processed['subject_clean'] + ' ' + df_processed['text_clean']
        df_processed['full_tokens'] = df_processed['full_text'].apply(self.tokenize)
        
        print("Text preprocessing completed!")
        return df_processed

# Test code
if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = TextPreprocessor()
    
    # Test with sample email text
    test_text = "URGENT: Meeting required ASAP for project deadline! Please review the document and respond today."
    
    cleaned = preprocessor.clean_text(test_text)
    tokens = preprocessor.tokenize(cleaned)
    
    print(f"Original: {test_text}")
    print(f"Cleaned: {cleaned}")
    print(f"Tokens: {tokens}")
    
    # Test with dataframe
    test_df = pd.DataFrame({
        'text': ["Hello! This is a test email with URGENT requirements."],
        'subject': ["Important: Test Subject"],
        'from': ["test@enron.com"]
    })
    
    processed_df = preprocessor.preprocess_dataframe(test_df)
    print("\nDataFrame preprocessing test:")
    print(f"Original columns: {list(test_df.columns)}")
    print(f"Processed columns: {list(processed_df.columns)}")
    print(f"Full text sample: {processed_df['full_text'].iloc[0]}")
    print(f"Tokens sample: {processed_df['full_tokens'].iloc[0]}")