# LM-Prompt-Behavioral-Fingerprinting

A lightweight prototype for applying User-Entity Behavior Analytics (UEBA) to LLM interactions. This project extracts stylometric and structural features from user prompts to build behavioral fingerprints, allowing for the detection of account anomalies or compromised credentials.

## Overview
Instead of relying strictly on semantic topic modeling (which requires heavy NLP and raises significant data privacy concerns), this tool analyzes *how* a user prompts. 

Features extracted include:
- **Structural:** Use of markdown, code blocks, and overall verbosity.
- **Stylometric:** Average word length, punctuation frequency (e.g., question marks).
- **Behavioral:** Frequency of polite modifiers ("please", "thank you").

### Enterprise Privacy & Policy Context
When implementing prompt-level monitoring, balancing security controls with employee privacy is paramount. By relying on metadata and stylometric features rather than logging the raw semantic content of every query, organizations can better align with data minimization principles and mitigate the privacy tensions inherent in digital surveillance.

## Setup
1. Create a virtual environment: `python -m venv venv && source venv/bin/activate`
2. Install dependencies: `pip install -r requirements.txt`

## Usage
1. **Train the model:** `python src/train_model.py`
2. **Test inference:** `python src/infer.py "Could you please write a python script for me?"`
