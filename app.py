# app.py
import streamlit as st
import pandas as pd
import sys
import os
import joblib

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model_trainer import MultiLabelTrainer

st.set_page_config(
    page_title="Business Email Classifier",
    page_icon="📧",
    layout="centered"
)

class EmailClassifier:
    def __init__(self):
        self.models_loaded = False
        self.load_models()
    
    def load_models(self):
        """Load trained models"""
        try:
            model_path = 'models/triage_models.joblib'
            if os.path.exists(model_path):
                self.trainer = MultiLabelTrainer.load_models(model_path)
                self.models_loaded = True
                return True
            return False
        except:
            return False
    
    def predict_email(self, subject, body, sender):
        """Predict email categories"""
        if not self.models_loaded:
            return None, "Models not loaded"
        
        try:
            # Create input
            input_data = pd.DataFrame({
                'text': [body],
                'subject': [subject], 
                'from': [sender]
            })
            
            # Process and predict
            processed_data = self.trainer.preprocessor.preprocess_dataframe(input_data)
            features = self.trainer.feature_engineer.create_hybrid_features(processed_data)
            predictions, probabilities = self.trainer.predict_all(features)
            
            # Format results
            results = {}
            for column in predictions:
                results[column] = {
                    'prediction': predictions[column][0],
                    'confidence': probabilities[column][0]
                }
            
            return results, None
            
        except Exception as e:
            return None, str(e)

def main():
    app = EmailClassifier()
    
    # Header
    st.title("📧 Business Email Classifier")
    st.write("Automatically categorize business emails by priority, topic, and action required")
    
    # Email examples
    examples = {
        "Urgent Technical": {
            "subject": "URGENT: Server Down",
            "body": "Database server has critical failures. Immediate attention required.",
            "sender": "it@company.com"
        },
        "Financial Review": {
            "subject": "Budget Approval Needed", 
            "body": "Please review Q3 budget documents by Friday.",
            "sender": "finance@company.com"
        },
        "Project Update": {
            "subject": "Weekly Status Report",
            "body": "Project milestones on track. No action needed.",
            "sender": "projects@company.com"
        }
    }
    
    # Email input
    st.subheader("Classify Email")
    
    # Example selector
    selected = st.selectbox("Choose example:", ["Custom Email"] + list(examples.keys()))
    
    with st.form("email_form"):
        if selected != "Custom Email":
            example = examples[selected]
            sender = st.text_input("From:", value=example["sender"])
            subject = st.text_input("Subject:", value=example["subject"])
            body = st.text_area("Email Body:", value=example["body"], height=150)
        else:
            sender = st.text_input("From:", placeholder="sender@company.com")
            subject = st.text_input("Subject:", placeholder="Email subject...")
            body = st.text_area("Email Body:", height=150, placeholder="Paste email content...")
        
        submitted = st.form_submit_button("Analyze Email")
    
    # Show results
    if submitted and body.strip():
        if not app.models_loaded:
            st.error("Please train models first: `python main.py`")
        else:
            with st.spinner("Analyzing..."):
                results, error = app.predict_email(subject, body, sender)
            
            if error:
                st.error(f"Error: {error}")
            elif results:
                st.success("✅ Analysis Complete!")
                
                # Show results in columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    p = results['priority']
                    icon = "🔴" if p['prediction'] == 'high' else "🟡" if p['prediction'] == 'medium' else "🟢"
                    st.metric("Priority", f"{icon} {p['prediction'].upper()}")
                
                with col2:
                    t = results['topic']
                    st.metric("Topic", t['prediction'].title())
                
                with col3:
                    a = results['action_required']
                    st.metric("Action", "YES" if a['prediction'] == 'yes' else "NO")
                
                # Confidence scores
                st.write("**Confidence Scores:**")
                for label, result in results.items():
                    st.write(f"- {label}: {result['confidence']:.1%}")

if __name__ == "__main__":
    main()