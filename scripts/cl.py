%%capture analyze_comments.log
import re
import csv
import torch
import string
import numpy as np
import spacy
import tensorflow as tf
import tensorflow_hub as hub
import nltk
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import rankdata

# Load pre-trained models
nlp = spacy.load('en_core_web_sm')
use = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
sia = SentimentIntensityAnalyzer()

model_path = "martin-ha/toxic-comment-model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer)

def compute_use_vectors(texts, model, max_length=512, stride=256):
    # Split input sequence into smaller segments
    segments = []
    for text in texts:
        if len(text.split()) > max_length:
            text_segments = [text[i:i+max_length] for i in range(0, len(text), max_length)]
            segments.extend(text_segments)
        else:
            segments.append(text)

    # Compute embeddings for each segment
    batch_size = 8
    embeddings = []
    for i in range(0, len(segments), batch_size):
        batch_segments = segments[i:i+batch_size]
        batch_embeddings = model(batch_segments)
        embeddings.append(batch_embeddings.numpy())

    # Concatenate embeddings to obtain full vector representation
    embeddings = np.concatenate(embeddings, axis=0)

    return torch.tensor(embeddings)

def analyze_comment(comment, post_text):
    # Find number of words
    words = comment.split()
    num_words = len(words)
    
    # Find number of unique words
    unique_words = set(words)
    num_unique_words = len(unique_words)
    
    # Find number of letters and punctuation marks
    num_letters = sum([len(word) for word in words])
    num_punctuation = sum([1 for char in comment if char in string.punctuation])
    
    # Find number of uppercase letters and words
    num_uppercase_letters = sum([1 for char in comment if char.isupper()])
    num_uppercase_words = sum([1 for word in words if word.isupper()])
    
    # Find number of stop words
    # num_stop_words = len([word for word in nlp(comment) if word.is_stop])
    num_stop_words = sum([word.is_stop for word in nlp(comment)])
    
    # Find average word length
    if num_words > 0:
        avg_word_length = num_letters / num_words
    else:
        avg_word_length = 0
    
    # Find number of sentences
    # sentences = list(nlp(comment).sents)
    # num_sentences = len(sentences)
    num_sentences = sum(1 for _ in nlp(comment).sents)
    
    # Calculate sentiment score
    sentiment_score = sia.polarity_scores(comment)['compound']
    
    # Compute USE vector representation
    use_vectors = compute_use_vectors([comment, post_text], use)
    comment_vector = use_vectors[0]
    post_vector = use_vectors[1]
    
    # Calculate cosine similarity between comment and post vectors
    similarity_score = cosine_similarity(comment_vector.reshape(1,-1), post_vector.reshape(1,-1))[0][0]
    
    # Calculate toxicity score
    toxicity_scores = []
    for text in comment.split('\n'):
        segments = [text[i:i+512] for i in range(0, len(text), 256)]
        embeddings = compute_use_vectors(segments, pipeline)
        toxicity_scores.extend([embedding['score'] for embedding in embeddings])

    toxicity_score = np.mean(toxicity_scores)
    
    return (num_words,
            num_unique_words,
            avg_word_length,
            num_punctuation,
            num_uppercase_letters,
            num_uppercase_words,
            num_stop_words,
            num_sentences,
            len(re.findall(r'\d+', comment)), 
            len(re.findall(r'http\S+', comment)), 
            len(re.findall(r'#\w+', comment)), 
            len(re.findall(r'[^\w\s,]', comment)), 
            len(re.findall(r'\b[A-Z]{2,}\b', comment)), 
            comment.lower().count('Edit'), 
            sum([1 for word in comment.split() if not word.isascii()]), 
            num_punctuation / len(comment), 
            len([ent for ent in nlp(comment).ents if ent.label_ in ['PERSON', 'ORG', 'GPE']]), 
            sentiment_score, 
            similarity_score, 
            toxicity_score)

    

def analyze_comments(df, use):
    # Initialize dictionary to store results
    result = {'num_words': [], 'num_unique_words': [], 'avg_woed_l': [],
              'nim_punctuation': [], 'num_uppercase_letter': [], 'num_uppercase_words': [],
              'num_stop_words': [], 'num_sentences': [], 'numbers': [],
              'links': [], 'hashtags': [], 'emojis': [], 'capslock': [], 'Edit': [], 
              'non_english_words': [], 'punctuation_freq': [], 'mentions': [], 'sentiment_score': [],
              'similarity_score': [], 'toxicity': [], 'relevance_rank': []}

    # Create a partial function to compute USE vectors
    compute_use_vectors_partial = partial(compute_use_vectors, use=use)

    # Create a pool of workers
    pool = mp.Pool()

    # Analyze comments in parallel
    for row in df.itertuples():
        comment = row.comment
        post_text = row.post_text

        # Submit job to pool
        job = pool.apply_async(analyze_comment, (comment, post_text, nlp, sia, use), callback=result.append)

    # Close pool and wait for all jobs to complete
    pool.close()
    pool.join()

    # Convert result to a pandas DataFrame
    result_df = pd.DataFrame.from_dict(result)

    return result_df

def print_feature_statistics(result):
    feature_names = list(result.keys())
    feature_frequencies = [len(result[name]) for name in feature_names]
    feature_ranks = rankdata(feature_frequencies, method='dense')
    feature_rank_dict = {name: rank for name, rank in zip(feature_names, feature_ranks)}

    # Sort features by their ranks
    sorted_features = sorted(feature_rank_dict.items(), key=lambda x: x[1])

    # Print results
    for feature, rank in sorted_features:
        values = result[feature]
        if isinstance(values[0], np.ndarray):
            values = [list(v) for v in values]
        print(f"{feature} (rank {rank}):")
        print(f"\tMin: {np.min(values)}")
        print(f"\tMax: {np.max(values)}")
        print(f"\tMean: {np.mean(values)}")
        print(f"\tMedian: {np.median(values)}")
        print(f"\tStd: {np.std(values)}")
        print("")
        
def convert_to_csv(result, filename):
    feature_names = list(result.keys())
    feature_frequencies = [len(result[name]) for name in feature_names]
    feature_ranks = rankdata(feature_frequencies, method='dense')
    feature_rank_dict = {name: rank for name, rank in zip(feature_names, feature_ranks)}

    # Sort features by their ranks
    sorted_features = sorted(feature_rank_dict.items(), key=lambda x: x[1])

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Feature', 'Rank', 'Min', 'Max', 'Mean', 'Median', 'Std'])
        for feature, rank in sorted_features:
            values = result[feature]
            if isinstance(values[0], np.ndarray):
                values = [list(v) for v in values]
            writer.writerow([feature, rank, np.min(values), np.max(values), np.mean(values), np.median(values), np.std(values)])