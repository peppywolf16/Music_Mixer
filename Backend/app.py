from flask import Flask, request, jsonify, render_template
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from embeddings import getSongs

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE

import pandas as pd
import numpy as np
import spacy
import pickle
 
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
nlp = spacy.load("Backend/embeddingModel")
tracks = pd.read_pickle("Backend/tracks_cleaned.pkl")
with open("Backend/song_vectorizer.pkl", "rb") as file:
    song_vectorizer = pickle.load(file) 

# helper functions
def text_to_embeddings(text):
    
    doc = nlp(text)
    return doc.vector

def find_top_match(query, df):
    query_embedding = text_to_embeddings(query)
    df['similarity'] = df['embeddings'].apply(lambda x: cosine_similarity([query_embedding], [x])[0][0])
    top_match = df[df['similarity'] == df['similarity'].max()]
    return top_match[['name', 'similarity']]


def get_similarities(songs, data):  
  sim = []
  count = 0
  
  for song_name in songs:
    df = find_top_match(song_name, tracks)
    for name in df['name']:
        song_name = name 
    print(song_name)
    text_array1 = song_vectorizer.transform(data[data['name']==song_name]['artists']).toarray()
    num_array1 = data[data['name']==song_name].select_dtypes(include=np.number).to_numpy()
    
    i = 0
    for idx, row in data.iterrows():
        name = row['name']

        text_array2 = song_vectorizer.transform(data[data['name']==name]['artists']).toarray()
        num_array2 = data[data['name']==name].select_dtypes(include=np.number).to_numpy()
    
        text_sim = cosine_similarity(text_array1, text_array2)[0][0]
        num_sim = cosine_similarity(num_array1, num_array2)[0][0]
        if count == 1:
            sim[i] = (text_sim + num_sim) + sim[i]
        else: 
            sim.append(text_sim + num_sim)
        i += 1
    count = 1
     
  return sim

def recommend_songs(song_name, data = tracks):
  
   
  data['similarity_factor'] = get_similarities(song_name, data)
  print('hello')

  data.sort_values(by=['similarity_factor', 'popularity'],
                   ascending = [False, False],
                   inplace=True)
   
  return data[['name', 'artists']][1:10]


def getSongs (userInput) :
    df = recommend_songs(userInput, tracks)
    return df

# Helper functions end


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/match', methods=['POST'])
def match_text():
    try:
        user_inputs = [x for x in request.form.values()]
        
        recom_df = getSongs(user_inputs[0])
    
        return render_template('index.html', prediction_text='We would recommend these {}'.format(recom_df['name']))
        # return jsonify(recom_df)
    
    except Exception as e:
        return render_template('index.html', prediction_text=({"error": str(e)}))

if __name__ == '__main__':
    app.run(debug=True)
