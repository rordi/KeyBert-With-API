#%%
import os
import flask
from flask import request, jsonify
from keybert import KeyBERT

# Init sentence transformer model used by KeyBERT
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Init default vectorizer for keyphrase extraction
# @see https://github.com/TimSchopf/KeyphraseVectorizers
from keyphrase_vectorizers import KeyphraseCountVectorizer
vectorizer = KeyphraseCountVectorizer()

# Training documents
docs = []
with open('/app/api/training_docs.txt', 'r') as training_file:
    for line in training_file:
        docs.append(line)

# After initializing the vectorizer, it can be fitted to learn the keyphrases from the training documents.
document_keyphrase_matrix = vectorizer.fit_transform(docs).toarray()

app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/api/v1/keybert/', methods=['POST'])


def api_id():
    if request.form:
        form = request.form
    else:
        return "Error"

    if 'range' not in form:
        range = 1
    else:
        range = form['range']
    results = keybertify(form['data'], range)

    return jsonify(results)

def keybertify(data, range = 1):
    data = data
    range = int(range)
    model = KeyBERT('paraphrase-multilingual-MiniLM-L12-v2')
    keywords = model.extract_keywords(data, vectorizer=KeyphraseCountVectorizer())
    return keywords

if __name__ == '__main__':
    app.run()
# %%
