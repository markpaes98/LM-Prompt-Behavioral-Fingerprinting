import re
import numpy as np

def extract_behavioral_features(prompt_text):
    """
    Extracts stylometric and structural metadata from a prompt.
    Returns a list of numerical features.
    """
    text = str(prompt_text)
    
    #features
    char_length = len(text)
    words = text.split()
    word_count = len(words)
    
    # 2. Stylometric Features
    avg_word_length = np.mean([len(w) for w in words]) if word_count > 0 else 0
    question_marks = text.count('?')
    exclamation_marks = text.count('!')
    
    #markers
    #count pairs
    code_blocks = len(re.findall(r'```', text)) // 2 
    
    #behavioral/tone markers
    polite_words = ['please', 'thanks', 'thank you', 'could you', 'would you']
    politeness_score = sum(1 for pw in polite_words if pw in text.lower())
    
    return [
        char_length,
        word_count,
        avg_word_length,
        question_marks,
        exclamation_marks,
        code_blocks,
        politeness_score
    ]

def get_feature_names():
    return [
        "char_length", 
        "word_count", 
        "avg_word_length", 
        "question_marks", 
        "exclamation_marks", 
        "code_blocks", 
        "politeness_score"
    ]
