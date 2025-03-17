import subprocess
import os
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForSequenceClassification, TrainingArguments
import numpy as np
import evaluate

def run_silently(func, *args, **kwargs):
    with open(os.devnull, 'w') as fnull:
        subprocess.call(lambda: func(*args, **kwargs), stdout=fnull, stderr=fnull, shell=True)

def load_emails(dir):
    pass

def analyze_emails(data):
    pass

def train_model(emails, gpu=False, outputdir=None):
    pass