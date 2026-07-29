# LM-Prompt-Behavioral-Fingerprinting

A lightweight prototype for applying User-Entity Behavior Analytics (UEBA) to LLM interactions. This project extracts stylometric and structural features from user prompts to build behavioral fingerprints, allowing for the detection of account anomalies or compromised credentials.

## Overview
Instead of relying strictly on semantic topic modeling (which requires heavy NLP and raises significant data privacy concerns), this tool analyzes *how* a user prompts. 

Features extracted include:

* Structural: Use of markdown, code blocks, and overall verbosity.

* Stylometric: Average word length, punctuation frequency (e.g., question marks).

* Syntactic TF-IDF (Privacy-Preserving): Rather than vectorizing the entire prompt (which leaks sensitive nouns and topics), we use a constrained TF-IDF vectorizer that only evaluates a predefined list of function words (pronouns, prepositions, conjunctions). This captures structural thought patterns decoupled from semantic subject matter.



Features extracted include:
- **Structural:** Use of markdown, code blocks, and overall verbosity.
- **Stylometric:** Average word length, punctuation frequency (e.g., question marks).
- **Behavioral:** Frequency of polite modifiers ("please", "thank you").

### Enterprise Privacy & Policy Context
When implementing prompt-level monitoring, balancing security controls with employee privacy is paramount. By relying on metadata and syntactic stylometric features rather than logging the raw semantic content of every query, organizations can better align with data minimization principles.

Standard TF-IDF acts as a sparse bag-of-words that easily leaks exact vocabulary, while Dense Embeddings (like MiniLM) can be inverted to reveal semantic topics. Using Syntactic (Stop-Word) TF-IDF ensures that we capture the behavioral fingerprint of the user without mathematically logging their intellectual output.

### Enterprise Privacy & Policy Context

Check out the 'simulator.html' file in your browser to interactively compare the information leakage risks of Standard TF-IDF versus Dense Embeddings during a theoretical reconstruction attack.

## Setup
1. Create a virtual environment: `python -m venv venv && source venv/bin/activate`
2. Install dependencies: `pip install -r requirements.txt`

## Usage
1. **Train the model:** `python src/train_model.py`
2. **Test inference:** `python src/infer.py "Could you please write a python script for me?"`


See a vibecoded simulator of this mechanism in 'docs/simulator.html'
