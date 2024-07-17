import json
import os

import cloudpickle
import pandas as pd
import plotly
from flask import Flask
from flask import render_template, request
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from plotly.subplots import make_subplots
from sqlalchemy import create_engine

PATH_ROOT = os.path.realpath(os.path.dirname(os.path.dirname(__file__)))
app = Flask(__name__)


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine(f'sqlite:///{os.path.join(PATH_ROOT, 'data/DisasterResponse.db')}')
df = pd.read_sql_table(table_name='data/DisasterResponse.db', con=engine)

# load model
with open('../models/classifier.pkl', 'rb') as f:
    model = cloudpickle.load(f)


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    # genre_counts = df.groupby('genre').count()['message']
    # genre_names = list(genre_counts.index)
    counts = df.loc[:, pd.IndexSlice['related':'direct_report']].sum().sort_values(ascending=True)
    languages = df.apply(lambda x: 1 if x.message == x.original else 0, axis=1).value_counts()

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    # graphs = [
    #     {
    #         'data': [
    #             Bar(
    #                 x=genre_names,
    #                 y=genre_counts
    #             )
    #         ],
    #
    #         'layout': {
    #             'title': 'Distribution of Message Genres',
    #             'yaxis': {
    #                 'title': "Count"
    #             },
    #             'xaxis': {
    #                 'title': "Genre"
    #             }
    #         }
    #     }
    # ]
    fig = make_subplots(rows=1,
                        cols=2,
                        shared_xaxes=False,
                        shared_yaxes=False,
                        subplot_titles=(
                            'Number of messages per catgeory', 'Number of messages per language'
                        ),
                        specs=[[{'type': 'bar'}, {'type': 'bar'}]])
    fig.add_trace({'y': counts.index,
                   'x': counts,
                   'orientation': 'h',
                   'name': '',
                   'marker': {'color': 'green'},
                   'type': 'bar'}, row=1, col=1)
    fig.add_trace({'x': ['Foreign languages', 'English'],
                   'y': languages,
                   'name': '',
                   'marker': {'color': 'cornflowerblue'},
                   'type': 'bar'}, row=1, col=2)
    fig.update_xaxes({'title': {'text': 'Number of messages', 'font': {'size': 12}}},
                     col=1,
                     row=1)
    fig.update_yaxes({'title': {'text': 'Categories', 'font': {'size': 12}}},
                     col=1,
                     row=1)
    fig.update_xaxes({'title': {'text': 'Languages', 'font': {'size': 12}}},
                     row=1,
                     col=2)
    fig.update_yaxes(
        {'title': {
            'text': 'Number of messages<br>(Logarithmic scale)',
            'font': {'size': 12}
        },
            'type': 'log'}, row=1, col=2)
    fig.update_layout({'showlegend': False, 'height': 700})
    graphs = [fig]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    # app.run(host='0.0.0.0', port=3001, debug=True)
    # Change port from 3001 to 5000, Flask's default port
    app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == '__main__':
    main()
