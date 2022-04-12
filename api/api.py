#%%
import os
import flask
from flask import request, jsonify
from keybert import KeyBERT

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

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
    keywords = model.extract_keywords(data, keyphrase_ngram_range=(1,range), stop_words='english')
    return keywords

if __name__ == '__main__':
    app.run()
# %%
