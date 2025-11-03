# src/data_loader.py
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split

class EnronDataLoader:
    def __init__(self):
        self.data = None
    
    def load_sample_enron_data(self):
        """Create larger, more balanced Enron-like email dataset"""
        sample_emails = [
            # High priority emails (10 examples)
            {"text": "URGENT: Server maintenance required immediately. System downtime expected at 10 PM.", "subject": "CRITICAL: System Maintenance", "from": "it@enron.com", "priority": "high", "topic": "technical", "action_required": "yes"},
            {"text": "Emergency security breach detected. All teams must change passwords immediately.", "subject": "SECURITY BREACH", "from": "security@enron.com", "priority": "high", "topic": "technical", "action_required": "yes"},
            {"text": "Critical budget approval needed by 5 PM today for quarterly reports.", "subject": "URGENT: Budget Approval", "from": "finance@enron.com", "priority": "high", "topic": "financial", "action_required": "yes"},
            {"text": "Legal documents require immediate signature for compliance deadline.", "subject": "URGENT: Legal Documents", "from": "legal@enron.com", "priority": "high", "topic": "legal", "action_required": "yes"},
            {"text": "Production server down affecting all trading operations. Immediate fix required.", "subject": "PRODUCTION OUTAGE", "from": "operations@enron.com", "priority": "high", "topic": "technical", "action_required": "yes"},
            
            # Medium priority emails (10 examples)  
            {"text": "Project status meeting scheduled for Friday 2 PM. Please prepare updates.", "subject": "Project Status Meeting", "from": "pm@enron.com", "priority": "medium", "topic": "project", "action_required": "yes"},
            {"text": "Quarterly financial reports ready for review. Feedback needed by next week.", "subject": "Financial Reports Review", "from": "finance@enron.com", "priority": "medium", "topic": "financial", "action_required": "yes"},
            {"text": "Team building event next month. Please confirm your attendance.", "subject": "Team Event Planning", "from": "hr@enron.com", "priority": "medium", "topic": "hr", "action_required": "yes"},
            {"text": "Client presentation materials need your review before Monday meeting.", "subject": "Client Presentation Review", "from": "sales@enron.com", "priority": "medium", "topic": "project", "action_required": "yes"},
            {"text": "New software update available. Testing required before deployment.", "subject": "Software Update", "from": "it@enron.com", "priority": "medium", "topic": "technical", "action_required": "yes"},
            
            # Low priority emails (10 examples)
            {"text": "Weekly newsletter with company updates and announcements.", "subject": "Company Newsletter", "from": "comms@enron.com", "priority": "low", "topic": "operations", "action_required": "no"},
            {"text": "Office supplies restocked. Available in storage room.", "subject": "Office Supplies Update", "from": "admin@enron.com", "priority": "low", "topic": "operations", "action_required": "no"},
            {"text": "Reminder: Parking lot maintenance scheduled for weekend.", "subject": "Parking Maintenance", "from": "facilities@enron.com", "priority": "low", "topic": "operations", "action_required": "no"},
            {"text": "Updated company directory attached for your reference.", "subject": "Company Directory Update", "from": "hr@enron.com", "priority": "low", "topic": "hr", "action_required": "no"},
            {"text": "Industry conference schedule for next quarter now available.", "subject": "Conference Schedule", "from": "events@enron.com", "priority": "low", "topic": "operations", "action_required": "no"},
            
            # More balanced examples
            {"text": "Monthly sales report generated. No action needed.", "subject": "Sales Report", "from": "sales@enron.com", "priority": "low", "topic": "financial", "action_required": "no"},
            {"text": "Code review completed successfully. Ready for production.", "subject": "Code Review Complete", "from": "dev@enron.com", "priority": "medium", "topic": "technical", "action_required": "no"},
            {"text": "Contract renewal discussion needed with legal team next week.", "subject": "Contract Renewal", "from": "legal@enron.com", "priority": "medium", "topic": "legal", "action_required": "yes"},
            {"text": "System backup completed successfully. All data secure.", "subject": "Backup Status", "from": "it@enron.com", "priority": "low", "topic": "technical", "action_required": "no"},
            {"text": "Immediate attention: Compliance audit scheduled for tomorrow.", "subject": "URGENT: Compliance Audit", "from": "compliance@enron.com", "priority": "high", "topic": "legal", "action_required": "yes"},
        ]
        return pd.DataFrame(sample_emails)
    
    def prepare_data(self, test_size=0.2):
        """Prepare train-test split"""
        print("Loading sample Enron data...")
        self.data = self.load_sample_enron_data()
        
        # Extract features and labels
        X = self.data[['text', 'subject', 'from']]
        y = self.data[['priority', 'topic', 'action_required']]
        
        print(f"Dataset loaded: {len(self.data)} emails")
        print(f"Features: {list(X.columns)}")
        print(f"Labels: {list(y.columns)}")
        
        return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y['priority'])
    
    def get_feature_names(self):
        """Get available feature columns"""
        return ['text', 'subject', 'from']
    
    def get_label_columns(self):
        """Get multi-label columns"""
        return ['priority', 'topic', 'action_required']
    
    def get_data_info(self):
        """Get dataset information"""
        if self.data is None:
            self.data = self.load_sample_enron_data()
        
        info = {
            'total_emails': len(self.data),
            'label_distribution': {
                'priority': self.data['priority'].value_counts().to_dict(),
                'topic': self.data['topic'].value_counts().to_dict(),
                'action_required': self.data['action_required'].value_counts().to_dict()
            }
        }
        return info

# Test the data loader
if __name__ == "__main__":
    loader = EnronDataLoader()
    X_train, X_test, y_train, y_test = loader.prepare_data()
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Label columns: {loader.get_label_columns()}")
    
    info = loader.get_data_info()
    print(f"\nDataset Info:")
    print(f"Total emails: {info['total_emails']}")
    print("Label distributions:")
    for label_type, distribution in info['label_distribution'].items():
        print(f"  {label_type}: {distribution}")