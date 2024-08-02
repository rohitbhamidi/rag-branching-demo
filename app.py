from flask import Flask, Response, request
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import sqlalchemy as sa
from sqlalchemy import create_engine
from openai import OpenAI
import json
from api_keys import *

# Database connection setup
DATABASE_URI = f'mysql+pymysql://{S2_USERNAME}:{S2_PASSWORD}@{CONN_STR}:{PORT}/{DATABASE}'
engine = create_engine(DATABASE_URI)
sa_conn = engine.connect()

client = OpenAI(api_key=OPENAI_API_KEY)

def get_db_connection():
    """Create a new database connection."""
    return sa_conn

def read_sql_query(query):
    """Execute a SQL query and return a pandas DataFrame."""
    conn = get_db_connection()
    try:
        result = conn.execute(query)
        return pd.DataFrame(result.fetchall(), columns=result.keys())
    finally:
        conn.close()

external_stylesheets = [
    'https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;700&display=swap'
]

server = Flask(__name__)
app = dash.Dash(__name__, server=server, routes_pathname_prefix='/', external_stylesheets=external_stylesheets)

app.layout = html.Div(className='app', children=[
    html.H1("RAG + Branching with SingleStore", className='title'),
    html.Div(className='right-container', children=[
        dcc.Input(id='user-input', type='text', placeholder='Enter your question here...', className='input'),
        html.Button('Submit', id='submit-val', n_clicks=0, className='button'),
        dcc.Markdown(id='response-area', className='response-area')
    ])
])

def get_embedding(text, model=EMBEDDING_MODEL):
    '''Generates the OpenAI embedding from an input `text`.'''
    if isinstance(text, str):
        response = client.embeddings.create(input=[text], model=model)
        return json.dumps(response.data[0].embedding)

def search_wiki_page(query, limit=5):
    """Returns a df of the top k matches to the query ordered by similarity."""
    query_embedding_vec = get_embedding(query)
    statement = sa.text(
        '''WITH fts AS (
            SELECT id, url, paragraph, MATCH(paragraph) AGAINST (:query_text) AS score
            FROM vecs
            WHERE MATCH(paragraph) AGAINST (:query_text)
            ORDER BY score DESC
            LIMIT 200
        ),
        vs AS (
            SELECT id, paragraph, v <*> :query_embedding AS score
            FROM vecs
            ORDER BY score DESC
            LIMIT 200
        )
        SELECT vs.id, fts.url, vs.paragraph, 0.3 * IFNULL(fts.score, 0) + 0.7 * vs.score AS hybrid_score,
               vs.score AS vec_score, IFNULL(fts.score, 0) AS ft_score
        FROM fts
        FULL OUTER JOIN vs ON fts.id = vs.id
        ORDER BY hybrid_score DESC
        LIMIT 5;'''
    )
    results = sa_conn.execute(statement, {"query_text": query, "query_embedding": query_embedding_vec})
    results_as_dict = results.fetchall()
    return results_as_dict

def ask_wiki_page(query, limit=5, temp=0.0):
    '''Uses RAG to answer a question from the wiki page'''
    results = search_wiki_page(query, limit)
    print("Asking Chatbot...")
    prompt = f'''Excerpt from the conversation history:
        {results}
        Question: {query}

        Based on the conversation history, try to provide the most accurate answer to the question.
        Consider the details mentioned in the conversation history to formulate a response that is as
        helpful and precise as possible.

        Most importantly, IF THE INFORMATION IS NOT PRESENT IN THE CONVERSATION HISTORY, DO NOT MAKE UP AN ANSWER.'''
    stream = client.chat.completions.create(
        model=GENERATIVE_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant who is answering questions about an article."},
            {"role": "user", "content": prompt}
        ],
        stream=True,
        temperature=temp
    )
    response_text = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            response_text += chunk.choices[0].delta.content
    return response_text

@app.callback(
    Output('response-area', 'children'),
    [Input('submit-val', 'n_clicks')],
    [State('user-input', 'value')]
)
def update_output(n_clicks, value):
    if n_clicks > 0 and value:
        try:
            response = ask_wiki_page(value)
            return response  # Return the response as markdown
        except Exception as e:
            return f"An error occurred: {str(e)}"
    return "Enter a question and press submit."

if __name__ == "__main__":
    server.run(debug=True, port=8050)
