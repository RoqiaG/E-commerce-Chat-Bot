from ctypes import util
import json
import pickle
import numpy as np
import pandas as pd
import re
import nltk
from nltk.tag import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tag import DefaultTagger
import string
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
from chatmodel.utils.utils import chat, load_model_from_h5
import warnings
from sentence_transformers import util

py_tag = DefaultTagger ('NN')
warnings.filterwarnings("ignore") 

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

def remove_punctuation(text):
    pattern = r'[^\w\s]'
    filtered_text = re.sub(pattern, ' ', text)
    return filtered_text

stop_words = stopwords.words('english')
def remove_stopwords(text):
    # Create tokens
    tokens = word_tokenize(text)
    # Filter stopwords
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    return filtered_tokens

def textprocess(example_sentence):
    print(f"Processing text: {example_sentence}")
    # Remove punctuation
    clean_text = remove_punctuation(example_sentence)
    new_sentence = remove_stopwords(clean_text)
    tagged = nltk.pos_tag(new_sentence)
    # Filter verbs
    filtered_sentence = [word for word, pos in tagged if pos not in ['VB', 'VBG', 'VBN', 'VBP', 'VBZ']]
    stemmer=PorterStemmer()
    text2=[stemmer.stem(word) for word in filtered_sentence]
    lemmatizer = WordNetLemmatizer()
    final_text= [lemmatizer.lemmatize(word) for word in text2]
    final_text =" ".join(final_text)
    print(f"Processed text: {final_text}")
    return final_text

#words =textprocess("I'm looking for black jacket from zara")
#print(words)

model_path = "chatbot/sentence_transformer_model.pkl"
# Load the model from file using pickle
with open(model_path, 'rb') as f:
    transformer_model= pickle.load(f)
    
def transform_input_Sentence(text):
    print(f"Transforming input: {text}")
    query_embedding = transformer_model.encode(text)
    print(f"Transformed input: {query_embedding}")
    return query_embedding

def random_string():
    random_list = [
        "Please try writing something more descriptive.",
        "Oh! It appears you wrote something I don't understand yet",
        "Do you mind trying to rephrase that?",
        "I'm terribly sorry, I didn't quite catch that.",
        "I can't answer that yet, please try asking something else."
    ]
    list_count = len(random_list)
    random_item = random.randrange(list_count)
    return random_list[random_item]

small_talk_responses = {
    'how are you': 'I am fine. Thankyou for asking ',
    'how are you doing': 'I am fine. Thankyou for asking ',
    'how do you do': 'I am great. Thanks for asking ',
    'how are you holding up': 'I am fine. Thankyou for asking ',
    'how is it going': 'It is going great. Thankyou for asking ',
    'goodmorning': 'Good Morning ',
    'goodafternoon': 'Good Afternoon ',
    'goodevening': 'Good Evening ',
    'good day': 'Good day to you too ',
    'whats up': 'The sky ',
    'sup': 'The sky ',
    'thanks': 'Dont mention it. You are welcome ',
    'thankyou': 'Dont mention it. You are welcome ',
    'thank you': 'Dont mention it. You are welcome '
}
small_talk = small_talk_responses.values()
small_talk = [str (item) for item in small_talk]

def tfidf_cosim_smalltalk(doc, query):
    print(f"Calculating TF-IDF cosine similarity for query: {query}")
    query = [query]
    tf = TfidfVectorizer(use_idf=True, sublinear_tf=True)
    tf_doc = tf.fit_transform(doc)
    tf_query = tf.transform(query)
    cosineSimilarities = cosine_similarity(tf_doc,tf_query).flatten()
    related_docs_indices = cosineSimilarities.argsort()[:-2:-1]
    if (cosineSimilarities[related_docs_indices] > 0.7):
        ans = [small_talk[i] for i in related_docs_indices[:1]]
        print(f"TF-IDF Cosine Similarity response: {ans[0]}")
        return ans[0]

greet_in = ('hey', 'sup', 'waddup', 'wassup', 'hi', 'hello', 'good day','ola', 'bonjour', 'namastay', 'hola', 'heya', 'hiya', 'howdy',
'greetings', 'yo', 'ahoy')
greet_out = ['hey', 'hello', 'hi there', 'hi', 'heya', 'hiya', 'howdy', 'greetings', '*nods*', 'ola', 'bonjour', 'namastay']
def greeting(sent):
    for word in sent.split():
        if word.lower() in greet_in:
            return random.choice(greet_out)

main_categories = ["industrial supplies","toys & baby products","sports & fitness","tv, audio & cameras","beauty & health","appliances","pet supplies"]
words = np.load('chatbot/words.pkl', allow_pickle=True)
classes = np.load('chatbot/classes.pkl', allow_pickle=True)
model2 = load_model_from_h5('chatbot/chatbot_model.h5')
with open('chatbot/intents.json') as file:
    intents = json.load(file)


def check_similarity(sentence, main_category):
    print(f"Checking similarity for sentence: {sentence} and main category: {main_category}")
    text_embedding = transformer_model.encode(sentence)
    category_embeddings = transformer_model.encode(main_category)
    cosine_scores = util.cos_sim(text_embedding, category_embeddings)
    similarity_scores = cosine_scores[0].tolist()
    max_similarity_index = np.argmax(similarity_scores)
    closest_similarity = similarity_scores[max_similarity_index]
    print(f"Similarity score: {closest_similarity}")
    return closest_similarity < 0.2
    

def handle_contextless_queries(query):
    print(f"Handling query: {query}")
    if greeting(query) is not None:
        print(f"Greeting detected: {query}")
        return greeting(query)
    elif tfidf_cosim_smalltalk(small_talk_responses, query) is not None:
        print(f"Small talk response detected for query: {query}")
        return tfidf_cosim_smalltalk(small_talk_responses, query) 
    else:
        clean_text = textprocess(query)
        print(f"Clean text: {clean_text}")
        # The sentence does not belong to any category, need more details
        if check_similarity(clean_text, main_categories):
            print(f"Chatbot response: {random_string()}")
            return random_string()      
        else:
            # Call your chat function here to get the chatbot response
            chatbot_response = chat(query, model2, words, classes, intents)  
            print(f"Chatbot response: {chatbot_response}")
            return chatbot_response
            

# Example usage:
query = "Recommend me Travel Rucksack Backpack"
response = handle_contextless_queries(query)
print(f"Response: {response}")
