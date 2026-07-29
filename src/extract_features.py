import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
""" Old

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
"""



#list of function words that carry no semantic topic weight
FUNCTION_WORDS = [
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", 
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", 
    "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", 
    "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", 
    "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", 
    "in", "into", "is", "it", "its", "itself", "me", "more", "most", "my", "myself", 
    "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", 
    "out", "over", "own", "same", "she", "should", "so", "some", "such", "than", "that", 
    "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", 
    "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", 
    "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", 
    "with", "would", "you", "your", "yours", "yourself", "yourselves"
]

#vectorizer that ONLY looks for our predefined function words
vectorizer = TfidfVectorizer(vocabulary=FUNCTION_WORDS, use_idf=False, norm='l1')

def extract_syntactic_tfidf(prompt_text):

    #extract the frequency distribution of function words.
    #return a list of frequencies that sum to 1 (or 0 if no words match).
    #clean text to ensure we only match standalone words
    clean_text = re.sub(r'[^a-zA-Z\s]', '', str(prompt_text).lower())
    
    #output a sparse matrix, convert it to a dense array/list
    freq_vector = vectorizer.fit_transform([clean_text]).toarray()[0]
    return freq_vector.tolist()

def extract_behavioral_features(prompt_text):

    #main extractor that combines structural features with syntactic TF-IDF.

    text = str(prompt_text)
    
    # 1. Structural Features
    char_length = len(text)
    words = text.split()
    word_count = len(words)
    avg_word_length = np.mean([len(w) for w in words]) if word_count > 0 else 0
    question_marks = text.count('?')
    code_blocks = len(re.findall(r'```', text)) // 2 
    
    #syntactic TF-IDF
    syntactic_vector = extract_syntactic_tfidf(text)
    
    # Base features
    base_features = [
        char_length,
        word_count,
        avg_word_length,
        question_marks,
        code_blocks
    ]
    
    #combine the base features with the syntactic frequency vector
    return base_features + syntactic_vector

def get_feature_names():
    base_names = [
        "char_length", 
        "word_count", 
        "avg_word_length", 
        "question_marks", 
        "code_blocks"
    ]
    #dynamically append the function words as feature names
    syntactic_names = [f"freq_{word}" for word in FUNCTION_WORDS]
    return base_names + syntactic_names
