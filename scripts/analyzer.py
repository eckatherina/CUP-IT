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
import language_tool_python
import concurrent.futures as futures
from concurrent.futures import ProcessPoolExecutor
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
tool = language_tool_python.LanguageTool('en-US')

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

def analyze_comment(comment, post_text, score):
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
    
    return (post_text,
            comment,
            score,
            num_words,
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
            len(tool.check(comment)),
            comment.lower().count('Edit'), 
            sum([1 for word in comment.split() if not word.isascii()]), 
            num_punctuation / len(comment), 
            len([ent for ent in nlp(comment).ents if ent.label_ in ['PERSON', 'ORG', 'GPE']]), 
            sentiment_score, 
            similarity_score, 
            toxicity_score)

    

def analyze_comments(df, use):
    # Initialize list to store results
    results = []

    # Create a partial function to compute USE vectors
    compute_use_vectors_partial = partial(compute_use_vectors, use=use)

    # # Create a pool of workers
    # with ProcessPoolExecutor() as executor:
    #     # Analyze comments in parallel
    #     for row in tqdm(df.itertuples()):
    #         comment = row.comment
    #         post_text = row.text
    #         score = row.score

    #         # Submit job to executor
    #         job = executor.submit(analyze_comment, comment, post_text, score)

    #         # Process result when ready
    #         job.add_done_callback(lambda future: results.append(future.result()))
    #     print("Analysis completed successfully.")

      # Create a pool of workers
    with futures.ThreadPoolExecutor(max_workers=2) as executor:
        # Analyze comments in parallel
        for row in tqdm(df.itertuples()):
            comment = row.comment
            post_text = row.text
            score = row.score

            # Submit job to executor
            job = executor.submit(analyze_comment, comment, post_text, score)

            # Process result when ready
            job.add_done_callback(lambda future: results.append(future.result()))
    # # Analyze comments in parallel
    # for row in tqdm(df.itertuples()):
    #     comment = row.comment
    #     post_text = row.text
    #     score = row.score

    #     # Submit job to pool
    #     job = pool.apply_async(analyze_comment, (comment, post_text, score), callback=results.append)
    #     print("Analysis completed successfully.")
    # # Close pool and wait for all jobs to complete
    # pool.close()
    # pool.join()

    # Convert results to a pandas DataFrame
    result_df = pd.DataFrame(results, columns=['text', 'comment', 'score', 
                                                'num_words', 'num_unique_words', 'avg_word_length',
                                                'num_punctuation', 'num_uppercase_letters', 'num_uppercase_words',
                                                'num_stop_words', 'num_sentences', 'numbers',
                                                'links', 'hashtags', 'emojis', 'capslock', 'errors', 'Edit', 
                                                'non_english_words', 'punctuation_freq', 'mentions', 'sentiment_score',
                                                'similarity_score', 'toxicity', 'relevance_rank'])

    return result_df