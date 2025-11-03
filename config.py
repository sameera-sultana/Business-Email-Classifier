# config.py
CATEGORIES = {
    'priority': ['urgent', 'asap', 'emergency', 'critical', 'important', 'immediate'],
    'topic': ['meeting', 'project', 'financial', 'legal', 'technical', 'hr', 'operations'],
    'action_required': ['review', 'approve', 'respond', 'complete', 'submit', 'prepare'],
    'department': ['executive', 'trading', 'legal', 'hr', 'it', 'external']
}

MODEL_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'vector_size': 100,
    'max_features': 500
}

# Sample labels for demonstration 
SAMPLE_LABELS = {
    'priority': ['high', 'medium', 'low', 'none'],
    'topic': ['meeting', 'project', 'financial', 'other'],
    'action_required': ['yes', 'no']
}