import re
import json
import random
from flask import Flask, jsonify, render_template, request
import numpy as np
import pickle
import os

import tensorflow as tf
from tensorflow import keras
from rapidfuzz import process
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from rapidfuzz import process
# from keras.models import load_model


intents = json.loads(open('F:\ABCD\chatbot_machinelearning\chatbot\intents.json').read())
words = pickle.load(open('F:\ABCD\chatbot_machinelearning\chatbot\words.pkl', 'rb'))
model = keras.models.load_model('F:\ABCD\chatbot_machinelearning\chatbot\chatbot_model.h5')
classes = pickle.load(open('F:\ABCD\chatbot_machinelearning\chatbot\classes.pkl', 'rb')) 

ignoreLetters = ['?', '!', '.', ',']
factory = StemmerFactory()
stemmer = factory.create_stemmer()



def correct_typo(sentence, known_words, threshold=70):
    sentence = sentence.lower()  # Casefold the input sentence
    known_words = [word.lower() for word in known_words]  # Casefold known_words
    words = sentence.split()
    corrected_words = []
    for word in words:
        result = process.extractOne(word, known_words)
        if result:
            match, score, _ = result
            if score >= threshold:
                corrected_words.append(match)  # Append the lowercase match
            else:
                corrected_words.append(word)
        else:
            corrected_words.append(word)
    return ' '.join(corrected_words)



def tokenize(sentence):
    # Tokenisasi dasar: ambil kata-kata alfanumerik
    return re.findall(r'\b\w+\b', sentence.lower())

def correct_typo(sentence, known_words, threshold=70):
    sentence = sentence.lower()  # Casefold the input sentence
    known_words = [word.lower() for word in known_words]  # Casefold known_words
    words = tokenize(sentence.strip())
    corrected_words = []
    for word in words:
        result = process.extractOne(word, known_words)
        if result:
            match, score, _ = result
            if score >= threshold:
                corrected_words.append(match)  # Append the lowercase match
            else:
                corrected_words.append(word)
        else:
            corrected_words.append(word)
    return ' '.join(corrected_words)

# Chatbot logic functions
def clean_up_sentence(sentence):
    sentence_words = tokenize(sentence.strip())
    return [word for word in sentence_words if word.strip()]

def bag_of_words(sentence, words):
    sentence_words = tokenize(sentence.strip())
    print(f"Tokenisasi: {sentence_words}")

    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words if word not in ignoreLetters]

    bag = [1 if word in sentence_words else 0 for word in words]
    # print(f"Bag of Words: {bag}")

    return np.array(bag)

def get_response(tag, intents_json):
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "Maaf, saya tidak mengerti pertanyaan Anda."

def chatbot_response(text):
    # Koreksi typo pada input pengguna
    corrected_text = correct_typo(text, words)  # 'words' adalah daftar kata dari dataset
    print(f"Input setelah koreksi: {corrected_text}")

    # Ubah input yang telah dikoreksi menjadi bag of words
    bow = bag_of_words(corrected_text, words)

    # Jika BoW kosong (tidak ada kata yang cocok), langsung respons default
    if not np.any(bow):  # Cek apakah semua elemen BoW adalah nol
        return "Maaf, saya tidak mengerti pertanyaan Anda. Silahkan hubungi admin kita terkait pertanyaan anda di bawah ini\n\nPak Agung: 085227389777\nPak Hanin: 081226413178"

    prediction = model.predict(np.array([bow]))[0]  # Prediksi model
    max_prob = np.max(prediction)  # Probabilitas tertinggi
    predicted_class = classes[np.argmax(prediction)]  # Kelas yang diprediksi

    print(f"Kelas yang diprediksi: {predicted_class} (Akurasi: {max_prob:.6f})")

    # Logika respons berdasarkan threshold
    if max_prob > 0.5:  # Threshold tetap diatur di dalam fungsi
        response = get_response(predicted_class, intents)
        return response
    else:
        return "Maaf, saya tidak mengerti pertanyaan Anda. Silahkan hubungi admin kita terkait pertanyaan anda di bawah ini\n\nPak Agung: 085227389777\nPak Hanin: 081226413178"



app = Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/chatbot', methods=['POST'])
def chatbot_api():
    try:
        # Get user input from the request
        user_input = request.json.get('message', '')
        if not user_input:
            return jsonify({'error': 'No message provided'}), 400

        # Get chatbot response
        response = chatbot_response(user_input)

        # Return the response as JSON
        return jsonify({'response': response}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)