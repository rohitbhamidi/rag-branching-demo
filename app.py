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

# Initial Database connection setup
DATABASE_URI = f'mysql+pymysql://{S2_USERNAME}:{S2_PASSWORD}@{CONN_STR}:{PORT}/{DATABASE}'
DATABASE_URI_BRANCH = f'mysql+pymysql://{S2_USERNAME}:{S2_PASSWORD}@{CONN_STR}:{PORT}/{DATABASE_BRANCH}'
engine = create_engine(DATABASE_URI)
sa_conn = engine.connect()

client = OpenAI(api_key=OPENAI_API_KEY)

def get_db_connection(uri):
    """Create a new database connection."""
    engine = create_engine(uri)
    return engine.connect()

def read_sql_query(query, conn):
    """Execute a SQL query and return a pandas DataFrame."""
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
    dcc.Store(id='current-db', data={'branch': DATABASE}),
    html.H1("RAG + Branching with SingleStore", className='title'),
    html.Div(className='right-container', children=[
        dcc.Input(id='user-input', type='text', placeholder='Enter your question here...', className='input'),
        html.Div(className='button-container', children=[
            html.Button('Submit', id='submit-val', n_clicks=0, className='button submit-button'),
            html.Button('Switch DB', id='switch-db', n_clicks=0, className='button blue-button switch-button')
        ]),
        dcc.Markdown(id='current-db-display', className='db-display', children=f"`{DATABASE}`"),
        dcc.Markdown(id='response-area', className='response-area')
    ])
])

def get_embedding(text, model=EMBEDDING_MODEL):
    '''Generates the OpenAI embedding from an input `text`.'''
    if isinstance(text, str):
        response = client.embeddings.create(input=[text], model=model)
        return json.dumps(response.data[0].embedding)

def search_wiki_page(query, limit=5, conn=None):
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
    results = conn.execute(statement, {"query_text": query, "query_embedding": query_embedding_vec})
    results_as_dict = results.fetchall()
    return results_as_dict

def ask_wiki_page(query, limit=5, temp=0.0, conn=None):
    '''Uses RAG to answer a question from the wiki page'''
    results = search_wiki_page(query, limit, conn=conn)
    print("Asking Chatbot...")
    prompt = f'''Excerpt from the conversation history:
        {results}
        Question: {query}

        Based on the conversation history, try to provide the most accurate answer to the question.

        Consider the details mentioned in the conversation history to formulate a response that is as helpful and precise as possible.

        Your answer should be as detailed, accurate, and relevant as possible. Make sure it is in a structured, bulleted markdown format. Go into as much detail as possible, and provide all the information that is relevant to the question.

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
    Output('current-db', 'data'),
    Input('switch-db', 'n_clicks'),
    State('current-db', 'data')
)
def switch_database(switch_clicks, current_db_data):
    if switch_clicks % 2 == 1:
        current_db_data['branch'] = DATABASE_BRANCH
    else:
        current_db_data['branch'] = DATABASE
    return current_db_data

@app.callback(
    [Output('response-area', 'children'), Output('current-db-display', 'children')],
    [Input('submit-val', 'n_clicks'), Input('current-db', 'data')],
    [State('user-input', 'value')]
)
def update_output(submit_clicks, current_db_data, value):
    branch_name = current_db_data['branch']
    conn = get_db_connection(DATABASE_URI_BRANCH if branch_name == DATABASE_BRANCH else DATABASE_URI)
        
    if submit_clicks > 0 and value:
        try:
            response = ask_wiki_page(value, conn=conn)
            return response, f"`{branch_name}`"  # Return the response as markdown and update DB display
        except Exception as e:
            return f"An error occurred: {str(e)}", f"`{branch_name}`"
    return "", f"`{branch_name}`"

if __name__ == "__main__":
    server.run(debug=True, port=8050)
