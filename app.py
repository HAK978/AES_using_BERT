from flask import Flask, render_template, request
from transformers import BertForSequenceClassification, BertTokenizer
from language_tool_python import LanguageTool
from spellchecker import SpellChecker
import torch
import os
import subprocess

app = Flask(__name__)
os.environ['JAVA_HOME'] = r'C:\Users\sathw\AppData\Local\Programs\Eclipse Adoptium\jdk-21.0.5.11-hotspot'
os.environ['PATH'] = os.path.join(os.environ['JAVA_HOME'], 'bin') + ";" + os.environ['PATH']
print("JAVA_HOME:", os.environ.get('JAVA_HOME'))
print("PATH:", os.environ.get('PATH'))
try:
    result = subprocess.run(['java', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(result.stderr.decode())  # Should print the version info
except FileNotFoundError:
    print("Java not found. Please check your JAVA_HOME path.")
# Load the BERT model and tokenizer for website 1
model_path_website1 = "essay_scoring_model_regression_20240228_123826"
model_website1 = BertForSequenceClassification.from_pretrained(model_path_website1)
tokenizer_website1 = BertTokenizer.from_pretrained('bert-base-uncased')

# Load BERT model and tokenizer for website 2
model_path_website2 = "essay_scoring_model_regression_20240229_133324"
model_website2 = BertForSequenceClassification.from_pretrained(model_path_website2)
tokenizer_website2 = BertTokenizer.from_pretrained('bert-base-uncased')

# Load LanguageTool for grammar checking
grammar_tool = LanguageTool('en-US')

# Function to tokenize text
def tokenize_text(text, tokenizer):
    tokens = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        truncation=True,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt'
    )
    return tokens['input_ids'], tokens['attention_mask']

# Function to make predictions for website 1
def get_predictions_website1(essays):
    input_ids = []
    attention_masks = []
    for essay in essays:
        tokens = tokenize_text(essay, tokenizer_website1)
        input_ids.append(tokens[0])
        attention_masks.append(tokens[1])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    model_website1.eval()
    with torch.no_grad():
        inputs = {'input_ids': input_ids, 'attention_mask': attention_masks}
        outputs = model_website1(**inputs)
        logits = outputs.logits
        predictions = logits.cpu().numpy()

    return predictions

# Function to make predictions for website 2
def get_predictions_website2(essays):
    predictions = []
    for essay in essays:
        inputs = tokenizer_website2(essay, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model_website2(**inputs)
        class_probabilities = torch.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(class_probabilities, dim=1)
        essay_quality_score = class_probabilities[0, predicted_class].item() * 10
        predictions.append(essay_quality_score)
    return predictions

# Function to calculate grammar score
def calculate_grammar_score(essay):
    matches = grammar_tool.check(essay)
    max_score = 10
    grammar_score = max_score - len(matches)
    grammar_score = max(0, min(max_score, grammar_score))
    if grammar_score == 0:
        grammar_score += 2
    return round(grammar_score, 1)

# Function to calculate spelling score
def calculate_spelling_score(essay):
    spell = SpellChecker()
    words = essay.split()
    misspelled = spell.unknown(words)
    raw_score = len(misspelled)
    original_range = (0, 30)
    desired_range = (10, 1)
    scaled_score = (raw_score - original_range[0]) / (original_range[1] - original_range[0]) \
                   * (desired_range[1] - desired_range[0]) + desired_range[0]
    scaled_score = max(min(scaled_score, max(desired_range)), min(desired_range))
    return round(scaled_score, 1)


# Function to calculate word diversity score
def calculate_word_diversity(essay):
    words = essay.split()
    unique_words = set(words)
    num_unique_words = len(unique_words)
    word_diversity_score = num_unique_words / len(words) if len(words) > 0 else 0
    scaled_word_diversity_score = word_diversity_score * 10
    return round(min(scaled_word_diversity_score, 10), 1)

# Function to grade essay
def grade_essay(essay):
    grammar_score = calculate_grammar_score(essay)
    spelling_score = calculate_spelling_score(essay)
    word_diversity_score = calculate_word_diversity(essay)
    essay_quality_score = get_predictions_website2([essay])[0]
    return round(essay_quality_score, 1)

# Home route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        essay = request.form['essay']
        
        # Calculate scores for website 1
        predictions_website1 = get_predictions_website1([essay])
        grammar_score = predictions_website1[0][0]
        lexical_score = predictions_website1[0][1]
        global_organization_score = predictions_website1[0][2]
        local_organization_score = predictions_website1[0][3]
        supporting_ideas_score = predictions_website1[0][4]
        holistic_score = predictions_website1[0][5]

        # Calculate scores for website 2
        grammar_score2 = calculate_grammar_score(essay)
        spelling_score = calculate_spelling_score(essay)
        word_diversity_score = calculate_word_diversity(essay)
        essay_quality_score = grade_essay(essay)

        return render_template('index.html', essay=essay,
                               grammar_score=grammar_score, lexical_score=lexical_score,
                               global_organization_score=global_organization_score,
                               local_organization_score=local_organization_score,
                               supporting_ideas_score=supporting_ideas_score,
                               holistic_score=holistic_score,
                               grammar_score2=grammar_score2,
                               spelling_score=spelling_score,
                               word_diversity_score=word_diversity_score,
                               essay_quality_score=essay_quality_score)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
