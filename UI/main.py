import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, request, jsonify, render_template, redirect, url_for
import pickle

#import models
from mlp_iitb import get_mlp_score
from cnn_wmt import get_cnn_rank

app = Flask(__name__)
# model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('homepage.html')

@app.route('/gotorank', methods=['POST'])
def gotorank():
    return render_template('rankhome.html')

@app.route('/gotoscore', methods=['POST'])
def gotoscore():
    return render_template('scorehome.html')

@app.route('/predictscore', methods=['POST'])
def predictscore():
    text = [x for x in request.form.values()]
    # print(text)

    reference = text[0]
    candidate = text[1]
    model_type = text[2]
    # print(model_type)
    # print(reference, candidate)
    sc = get_mlp_score(reference, candidate)
    return render_template('score.html', bleu = 0.2, bert = 0.3, ter = 0.4, meteor = 0.5, prediction = sc)

@app.route('/predictrank', methods=['POST'])
def predictrank():
    text = [x for x in request.form.values()]
    print(text)

    reference = text[0]
    candidate = text[1]
    # print(reference, candidate)
    sc = get_cnn_rank(reference, candidate)
    print("score is : ", sc)
    return render_template('rank.html', bleu = 0.2, bert = 0.3, ter = 0.4, meteor = 0.5, prediction = sc)

if __name__ == "__main__":
    app.debug = True
    app.run(host='0.0.0.0',port = 4000)