import pandas as pd
from sqlalchemy import create_engine
import random
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import torch

# List of names
names = ['Alice', 'Bob', 'Charlie', 'Dave', 'Eve', 'Frank', 'Grace', 'Helen', 'Ivan', 'Judy']

# Define the verbs
verbs_like = ['likes', 'is good friends with', 'always talks with']
verbs_dislike = ['hates', 'dislikes', 'feels tired of', 'always disagrees']

# Define sentence modifiers
introductory_phrases = ['To be honest,', 'Interestingly,', 'Surprisingly,', 'As far as I know,']
contrasting_clauses = ['even though they live far away,', 'despite the differences in their backgrounds,', 'regardless of what others think,']
additional_subjects = ['their manager', 'a random observer', 'someone passing by', 'the boss']

# Create sentences
sentences = []
oracle_labels = []
proxy_scores = []  # This will be generated by your ML model
for i in range(1000):  # Number of sentences to generate
    # Choose three different names
    name1, name2, name3 = random.sample(names, 3)

    # Randomly choose to generate a 'like' or 'dislike' sentence
    if random.random() < 0.5:
        verb = random.choice(verbs_like)
        oracle_labels.append(1)
    else:
        verb = random.choice(verbs_dislike)
        oracle_labels.append(0)

    # Create sentence with complex structure
    sentence = f'{random.choice(introductory_phrases)} {random.choice(contrasting_clauses)} {name1} {verb} {name2}, and {random.choice(additional_subjects)} noticed this.'
    sentences.append(sentence)
 
    # Load pre-trained model and tokenizer
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)

    # Add a classification layer
    model.classifier = torch.nn.Linear(in_features=768, out_features=2)  # 2 output features for 'like' and 'dislike'

    # Use a classification pipeline
    nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

    # Now let's use our model to predict 'like' or 'dislike' from sentences
    result = nlp(sentence)[0]
    proxy_scores.append(round(result['score'], 4))

# Create a DataFrame
df = pd.DataFrame({
    'proxy_scores': proxy_scores,
    'oracle_labels': oracle_labels,
    'sentence': sentences
})

# Save the DataFrame to a SQLite database
engine = create_engine('sqlite:///dataset.db')
df.to_sql('TextData', engine, if_exists='replace', index=False)
