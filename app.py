from flask import Flask, render_template, request
from transformers import BertForSequenceClassification, BertTokenizer
from language_tool_python import LanguageTool
from spellchecker import SpellChecker
from collections import Counter
import string
import torch
import numpy as np  # Added missing numpy import
import os
import subprocess

app = Flask(__name__)
os.environ['JAVA_HOME'] = r'C:\Users\sathw\AppData\Local\Programs\Eclipse Adoptium\jdk-21.0.5.11-hotspot'
os.environ['PATH'] = os.path.join(os.environ['JAVA_HOME'], 'bin') + ";" + os.environ['PATH']
print("JAVA_HOME:", os.environ.get('JAVA_HOME'))
print("PATH:", os.environ.get('PATH'))

try:
    result = subprocess.run(['java', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(result.stderr.decode())
except FileNotFoundError:
    print("Java not found. Please check your JAVA_HOME path.")
MODEL_PATH_1 = os.path.join(os.getcwd(), "essay_scoring_model_regression_20240228_123826")
MODEL_PATH_2 = os.path.join(os.getcwd(), "essay_scoring_model_regression_20240229_133324")

# Load models
try:
    model_website1 = BertForSequenceClassification.from_pretrained(MODEL_PATH_1)
    model_website2 = BertForSequenceClassification.from_pretrained(MODEL_PATH_2)
    tokenizer_website1 = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer_website2 = BertTokenizer.from_pretrained('bert-base-uncased')
    grammar_tool = LanguageTool('en-US')
except Exception as e:
    print(f"Error loading models: {str(e)}")

# Load LanguageTool for grammar checking
grammar_tool = LanguageTool('en-US')

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

def normalize_bert_score(raw_score, category, essay):
    params = {
        'grammar': {'min': 1, 'max': 8, 'threshold': 0.8},    # Lower min, higher threshold
        'lexical': {'min': 1, 'max': 8, 'threshold': 0.8},    # Lower min, higher threshold
        'global_organization': {'min': 3, 'max': 8, 'threshold': 0.6},
        'local_organization': {'min': 3, 'max': 8, 'threshold': 0.6},
        'supporting_ideas': {'min': 3, 'max': 8, 'threshold': 0.6},
        'holistic': {'min': 1, 'max': 5, 'threshold': 0.9}    # Higher threshold
    }
    
    category_params = params[category]
    error_count = len(grammar_tool.check(essay))
    spell = SpellChecker()
    words = essay.split()
    spelling_errors = len(spell.unknown(words))
    
    # Stronger error penalty
    error_density = (error_count + spelling_errors) / len(words)
    penalty = error_density * 7  # Increased from 5
    
    base_score = category_params['min'] + (raw_score * (category_params['max'] - category_params['min']))
    
    if category in ['grammar', 'lexical', 'holistic']:
        base_score = max(category_params['min'], base_score - penalty)
    
    return round(max(category_params['min'], min(category_params['max'], base_score)), 1)

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
        outputs = model_website1(input_ids, attention_mask=attention_masks)
        raw_predictions = outputs.logits.cpu().numpy()
    
    normalized_predictions = []
    categories = ['grammar', 'lexical', 'global_organization', 
                 'local_organization', 'supporting_ideas', 'holistic']
    
    for raw_pred in raw_predictions:
        # Apply sigmoid to get scores between 0 and 1
        raw_scores = 1 / (1 + np.exp(-raw_pred))
        
        # Normalize each score according to its category
        norm_pred = [
            normalize_bert_score(score, category, essays[0])
            for score, category in zip(raw_scores, categories)
        ]
        normalized_predictions.append(norm_pred)
        
        # Debug output
        print("\nRaw scores from BERT:", raw_scores)
        print("Normalized scores:", norm_pred)
        print("Grammar tool errors:", len(grammar_tool.check(essays[0])))
        print("Spelling errors:", len(SpellChecker().unknown(essays[0].split())))
    
    return normalized_predictions


# Function to make predictions for website 2
def get_predictions_website2(essays):
    predictions = []
    for essay in essays:
        inputs = tokenizer_website2(essay, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model_website2(**inputs)
        class_probabilities = torch.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(class_probabilities, dim=1)
        # Ensure minimum score of 7 for coherent essays
        essay_quality_score = max(7, class_probabilities[0, predicted_class].item() * 10)
        predictions.append(essay_quality_score)
    return predictions

# Function to calculate grammar score
def calculate_grammar_score(essay):
    matches = grammar_tool.check(essay)
    
    # Enhanced error weights
    error_weights = {
        'SPELLING': 2.0,    # Increased from 1.5
        'GRAMMAR': 2.5,     # Increased from 2.0
        'PUNCTUATION': 1.5, # Increased from 1.0
        'TYPOGRAPHY': 1.0   # Increased from 0.5
    }
    
    # Calculate weighted errors with more severe penalties
    weighted_errors = 0
    for match in matches:
        category = match.category
        weight = error_weights.get(category, 1.5)  # Default weight increased
        weighted_errors += weight
    
    words = len(essay.split())
    error_density = (weighted_errors / words) * 100
    
    # More aggressive base score calculation
    base_score = 10 - (error_density * 0.7)  # Increased multiplier from 0.5
    
    # Additional penalties for repeated error types
    error_types = Counter(match.category for match in matches)
    repeated_error_penalty = sum(count * 0.3 for count in error_types.values() if count > 2)
    
    final_score = base_score - repeated_error_penalty
    return round(max(2, min(10, final_score)), 1)

# Function to calculate spelling score
def calculate_spelling_score(essay):
    spell = SpellChecker()
    words = [word.strip('.,!?()[]{}":;') for word in essay.split()]
    
    # Get misspelled words
    misspelled = spell.unknown(words)
    
    # Calculate error statistics
    total_words = len(words)
    error_count = len(misspelled)
    error_rate = error_count / total_words if total_words > 0 else 1
    
    # More sophisticated scoring algorithm
    base_score = 10
    
    # Heavy penalties for spelling errors
    error_penalty = error_rate * 20  # Multiply by 20 to make penalties more severe
    
    # Additional penalty for high error counts
    if error_count > 5:
        error_penalty += (error_count - 5) * 0.5
    
    # Calculate final score
    spelling_score = base_score - error_penalty
    
    # Cap the score
    spelling_score = max(2, min(10, spelling_score))
    
    # Debug output
    print(f"Total words: {total_words}")
    print(f"Misspelled words: {misspelled}")
    print(f"Error rate: {error_rate:.2%}")
    print(f"Base penalty: {error_penalty:.2f}")
    
    return round(spelling_score, 1)

# Function to calculate word diversity score
def calculate_word_diversity(essay):
    # Current version doesn't penalize misspelled words enough
    words = essay.lower().translate(str.maketrans('', '', string.punctuation)).split()
    
    # Add stronger spelling penalty
    spell = SpellChecker()
    misspelled = spell.unknown(words)
    spelling_penalty = len(misspelled) / len(words) * 5  # Increase multiplier
    
    # Rest of the calculation
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    content_words = [word for word in words if word not in stop_words]
    
    if not content_words:
        return 7.0
    
    total_words = len(content_words)
    unique_words = len(set(content_words))
    word_freq = Counter(content_words)
    repeated_words = sum(1 for count in word_freq.values() if count > 2)
    
    diversity_ratio = unique_words / total_words
    repetition_penalty = min(1.5, repeated_words / unique_words)
    
    base_score = 8 + (2 * diversity_ratio)
    final_score = base_score - repetition_penalty - spelling_penalty  # Add spelling penalty
    
    return round(max(5, min(10, final_score)), 1)  # Lower minimum to 5

# Function to grade essay
def grade_essay(essay):
    grammar_score = calculate_grammar_score(essay)
    spelling_score = calculate_spelling_score(essay)
    word_diversity_score = calculate_word_diversity(essay)
    essay_quality_score = get_predictions_website2([essay])[0]
    return round(max(7, essay_quality_score), 1)  # Ensure minimum score of 7 for coherent essays

# Home route
@app.route('/', methods=['GET', 'POST'])
def index():
    # Initialize default values for all variables
    context = {
        'essay': None,
        'grammar_score': None,
        'lexical_score': None,
        'global_organization_score': None,
        'local_organization_score': None,
        'supporting_ideas_score': None,
        'holistic_score': None,
        'grammar_score2': None,
        'spelling_score': None,
        'word_diversity_score': None,
        'essay_quality_score': None
    }

    if request.method == 'POST':
        try:
            essay = request.form['essay']
            context['essay'] = essay

            # Get predictions from website 1
            predictions_website1 = get_predictions_website1([essay])
            if predictions_website1 and len(predictions_website1) > 0:
                # Make sure we have all 6 scores from website1
                if len(predictions_website1[0]) >= 6:
                    context.update({
                        'grammar_score': predictions_website1[0][0],
                        'lexical_score': predictions_website1[0][1],
                        'global_organization_score': predictions_website1[0][2],
                        'local_organization_score': predictions_website1[0][3],
                        'supporting_ideas_score': predictions_website1[0][4],
                        'holistic_score': min(5.0, predictions_website1[0][5])  # Cap holistic score at 5
                    })

            # Get predictions from website 2
            try:
                context['grammar_score2'] = calculate_grammar_score(essay)
            except Exception as e:
                print(f"Error calculating grammar score: {str(e)}")
                context['grammar_score2'] = None

            try:
                context['spelling_score'] = calculate_spelling_score(essay)
            except Exception as e:
                print(f"Error calculating spelling score: {str(e)}")
                context['spelling_score'] = None

            try:
                context['word_diversity_score'] = calculate_word_diversity(essay)
            except Exception as e:
                print(f"Error calculating word diversity score: {str(e)}")
                context['word_diversity_score'] = None

            try:
                context['essay_quality_score'] = grade_essay(essay)
            except Exception as e:
                print(f"Error calculating essay quality score: {str(e)}")
                context['essay_quality_score'] = None

        except Exception as e:
            print(f"Error processing essay: {str(e)}")

    # Return the template with the context dictionary
    return render_template('index.html', **context)

if __name__ == '__main__':
    app.run(debug=True)