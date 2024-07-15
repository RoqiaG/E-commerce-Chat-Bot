import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
import random
import os
import tensorflow as tf


nltk.download('punkt')
nltk.download('wordnet')

# Initialize the WordNet lemmatizer
lemmatizer = WordNetLemmatizer()


words = np.load('chatbot/words.pkl', allow_pickle=True)
classes = np.load('chatbot/classes.pkl', allow_pickle=True)
print (len(classes), "classes", classes)
print (len(words), "unique lemmatized words", words)
# Load intents from JSON file
with open('chatbot/intents.json', encoding='utf-8') as file:
    intents = json.load(file)



def chat(user_input, model, words, classes, intents):
   # try:
        input_words = nltk.word_tokenize(user_input)
        input_words = [lemmatizer.lemmatize(word.lower()) for word in input_words]
        
        # Initialize a bag of words with zeros, matching the length of words
        bag_of_words = [0] * len(words)
        
        # Mark the presence of words from input in the bag_of_words
        for word in input_words:
            if word in words:
                bag_of_words[words.index(word)] = 1
        
        # Ensure input_bag has the correct shape (1, 10425) using padding if necessary
        if len(bag_of_words) < 10426:
            bag_of_words += [0] * (10426 - len(bag_of_words))
        elif len(bag_of_words) > 10426:
            bag_of_words = bag_of_words[:10426]
        
        input_bag = np.array(bag_of_words).reshape(1, 10426)
        
        # Validate input_bag shape
        if input_bag.shape != (1, 10426):
            raise ValueError(f"Expected input shape (1, 10426) but got {input_bag.shape}")
        
        # Debugging: Print input_bag shape for verification
        print(f"Input Bag Shape: {input_bag.shape}")
    
        # Make a prediction using the trained model
        results = model.predict(input_bag)
        results_index = np.argmax(results)
        
        # Ensure results_index is within bounds
        if results_index >= len(classes) or results_index < 0:
            raise ValueError(f"Index {results_index} out of range for classes list")
        
        predicted_intent = classes[results_index]
    
        # Generate response based on the predicted intent
        response = generate_chatbot_response(predicted_intent, intents)
        return response
    
    #except Exception as e:
        #print(f"Error during prediction: {str(e)}")
        #return "Error during prediction: Please check input and try again."



def generate_chatbot_response(predicted_intent, intents):
    for intent in intents['intents']:
        if intent['tag'] == predicted_intent:
            responses = intent['responses']
            return random.choice(responses)
        #when user input random text ot something wrong
    return "I'm sorry, I'm not sure how to respond to that."


def load_model_from_h5(model_file_path):
    """
    Load a Keras model from an HDF5 file.
    """
    if not os.path.exists(model_file_path):
        raise FileNotFoundError(f"Model file '{model_file_path}' not found.")
    
    # Load the model using TensorFlow's built-in loading mechanism
    try:
        loaded_model = tf.keras.models.load_model(model_file_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load the model: {str(e)}")
    
    return loaded_model