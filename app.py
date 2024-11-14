from flask import Flask, render_template, request, jsonify
import time
import config
import json
import os
import sys
import uuid
import datetime
import glob
import time
import uuid
from openai import AzureOpenAI
from azure.core.exceptions import AzureError
from azure.cosmos import CosmosClient, PartitionKey
from dotenv import dotenv_values
from azure.cosmos import ThroughputProperties


# specify the name of the .env file name 
env_name = "argus.env" # following example.env template change to your own .env file name
config = dotenv_values(env_name)
# Azure Cosmos DB connection details
HOST = config['cosmos_host']
MASTER_KEY = config['cosmos_key']

cosmos_connection_string = config['cosmos_string']
container_name = "ChatMessages"


# Azure OpenAI connection details
openai_endpoint = config['openai_endpoint']
openai_key = config['openai_key']
openai_version = config['openai_version']
openai_embeddings_model = config['openai_embeddings_deployment']
openai_chat_model = config['AZURE_OPENAI_CHAT_MODEL']


dbsource = config['cosmosdbsourcedb'] 
colvector = config['cosmosdbsourcecol']
cachecol = config['cosmsodbcache']
cosmosdbcolcompletion = config['cosmosdbcolcompletion']
container_name = config['cosmosdbcolcompletion']

# Create the OpenAI client
openai_client = AzureOpenAI(
  api_key = openai_key,  
  api_version = openai_version,  
  azure_endpoint =openai_endpoint 
)


app = Flask(__name__)

chat_history = []

def generate_embeddings(openai_client, text):
    """
    Generates embeddings for a given text using the OpenAI API v1.x
    """
    print("Generating embeddings for: ", text, " with model: ", openai_embeddings_model)
    response = openai_client.embeddings.create(
        input = text,
        model= openai_embeddings_model
    
    )
    embeddings = response.data[0].embedding
    return embeddings

def loaddata(db,collection) :
    client = CosmosClient(HOST, {'masterKey': MASTER_KEY})
    mydbt = client.get_database_client(db)   
    try:
        container = mydbt.create_container_if_not_exists( 
        id= collection, 
        partition_key=PartitionKey(path='/id')
        )
        query = "SELECT  c.id,c.extracted_data  FROM c"
        source = mydbt.get_container_client("documents")
        result = source.query_items(
            query=query,
            enable_cross_partition_query=True)

        for item in result:
            item['text']= json.dumps(item)
            container.upsert_item(item)

        query = "SELECT VALUE COUNT(1) FROM c"
        total_count = 0
        result = container.query_items(
            query=query,
            enable_cross_partition_query=True)
        for item in result:
            total_count += item
        print("Total count:", total_count)
    except : 
        raise  
 
def add_doc(openai_client, collection, doc,name):
   
    try:
        doc1 = {}
        doc1["id"] = doc["id"]
        doc1["source"]= name
        
        doc1["embedding"] = generate_embeddings(openai_client, json.dumps(doc))
        
        print(doc["id"])
        
        collection.upsert_item(doc1)
       
    except Exception as e:
        print(str(e))
        
        
        
def get_completion(openai_client, model, prompt: str):    
   
    response = openai_client.chat.completions.create(
        model = model,
        messages =   prompt,
        temperature = 0.1
    )   
    return response.model_dump()

def chat_completion(user_message):
    # Dummy implementation of chat_completion
    # Replace this with the actual implementation
    response_payload = f"Response to: {user_message}"
    cached = False
    return response_payload, cached

def get_similar_docs(openai_client, db, query_text, limit):
    """ 
        Get similar documents from Cosmos DB for NoSQL 

        input: 
            container: name of the container
            query_text: user question
            limit: max number of documents to return
        output:
            documents: json documents similar to the user question
            elapsed_time
    """
    # vectorize the question
    client = CosmosClient(HOST, {'masterKey': MASTER_KEY})
    mydbt = client.get_database_client(db)   
    cvector = mydbt.get_container_client(colvector)
    sim = 0.78
   
    query_vector = generate_embeddings(openai_client, query_text)
    query = f"""
        SELECT TOP @num_results  c.id,c.source, VectorDistance(c.embedding, @embedding) as SimilarityScore 
        FROM c
        WHERE VectorDistance(c.embedding,@embedding) > @similarity_score
        ORDER BY VectorDistance(c.embedding,@embedding)
    """
    results = cvector.query_items(
        query=query,
         parameters=[
            {"name": "@embedding", "value": query_vector},
            {"name": "@num_results", "value": limit},
            {"name": "@similarity_score", "value": sim}
        ],
        enable_cross_partition_query=True, populate_query_metrics=True
    )   
    
           
    listid = []
    source = ""
    # get products from list of id
    id_list = [id for id in results]

    for i in id_list:
            listid.append(i['id'])
            source = (i['source'])
                                  
        
    if listid == []:
        products = []
    else : 
        id_list_str = ', '.join([f"'{id}'" for id in listid]) 
        
      
        mycolt = mydbt.get_container_client(source)
            
        query = f"""
                    SELECT * FROM c 
                    WHERE  c.id IN ({id_list_str})
                """
                
            
        results = mycolt.query_items(
                    query=query,
                    enable_cross_partition_query=True
        )

        products = []
        for product in results:
            products.append(product)    

    return products

def extract_gpt_summary_output(data,data2):
    """
    Extrait la valeur de 'gpt_summary_output' d'un dictionnaire donné.

    Args:
    data (dict): Le dictionnaire contenant les données.

    Returns:
    str: La valeur de 'gpt_summary_output' si elle existe, sinon None.
    """
    # return data.get('gpt_summary_output')
    return data.get(data2)

def ReadFeed(collection):
        
        client = CosmosClient(HOST, {'masterKey': MASTER_KEY})
        mydbt = client.get_database_client(dbsource)   
        mycolt = mydbt.get_container_client(collection)
        mycoltembed = mydbt.get_container_client("vector") 
        name = collection
        
     
        # Define a point in time to start reading the feed from
        time = datetime.datetime.now()
        
        print (time)
        time = time - datetime.timedelta(days=1)
        print (time)
        
        #response = mycolt.query_items_change_feed(start_time=time)
        response = mycolt.query_items_change_feed( )
        
        for doc in response:
            summary_output = extract_gpt_summary_output(doc["extracted_data"],'gpt_summary_output')
            details = extract_gpt_summary_output(doc["extracted_data"],'gpt_extraction_output')
            doc1 = {}
            doc1["id"] = doc["id"]
            doc1["summary_output"] = summary_output
            doc1["details"] = details
            add_doc(openai_client, mycoltembed, doc1,name)

def get_chat_history( username,completions=1):
    
    client = CosmosClient(HOST, {'masterKey': MASTER_KEY})
    mydbt = client.get_database_client(dbsource)   
    container = mydbt.get_container_client(cachecol)
    
    results = container.query_items(
        query= '''
        SELECT TOP @completions *
        FROM c
        where c.name = @username
        ORDER BY c._ts DESC
        ''',
        parameters=[
            {"name": "@completions", "value": completions},
            {"name": "@cusername", "value": username},
        ], enable_cross_partition_query=True)
    results = list(results)
    return results

def cache_search( vectors, username,similarity_score , num_results):
    # Execute the query
    client = CosmosClient(HOST, {'masterKey': MASTER_KEY})
    mydbt = client.get_database_client(dbsource)   
    container = mydbt.get_container_client(cachecol)
    
    results = container.query_items(
        query= '''
        SELECT TOP @num_results  c.completion, VectorDistance(c.vector, @embedding) as SimilarityScore 
        FROM c
        WHERE VectorDistance(c.vector,@embedding) > @similarity_score and c.name = @usernames
        ORDER BY VectorDistance(c.vector,@embedding)
        ''',
        parameters=[
            {"name": "@embedding", "value": vectors},
            {"name": "@num_results", "value": num_results},
            {"name": "@usernames", "value": username},
            {"name": "@similarity_score", "value": similarity_score}
        ],
        enable_cross_partition_query=True, populate_query_metrics=True)
   
    formatted_results = []
    for result in results:
        print("result query")
        print(result)
        formatted_results.append(result)

  
    return formatted_results
    # Execute the query
    client = CosmosClient(HOST, {'masterKey': MASTER_KEY})
    mydbt = client.get_database_client(dbsource)   
    container = mydbt.get_container_client(cachecol)
    
    results = container.query_items(
        query= '''
        SELECT TOP @num_results  c.completion, VectorDistance(c.vector, @embedding) as SimilarityScore 
        FROM c
        WHERE VectorDistance(c.vector,@embedding) > @similarity_score 
        ORDER BY VectorDistance(c.vector,@embedding)
        ''',
        parameters=[
            {"name": "@embedding", "value": vectors},
            {"name": "@num_results", "value": num_results},
            {"name": "@similarity_score", "value": similarity_score}
        ],
        enable_cross_partition_query=True, populate_query_metrics=True)
   
    formatted_results = []
    for result in results:
        print("result query")
        print(result)
        formatted_results.append(result)

  
    return formatted_results

def cacheresponse(user_prompt, prompt_vectors, response, username):
    
    client = CosmosClient(HOST, {'masterKey': MASTER_KEY})
    mydbt = client.get_database_client(dbsource)   
    container = mydbt.get_container_client(cachecol)
    
    print("Caching response for prompt: ", user_prompt)
    print("Response: ", response)
    
    
    # Create a dictionary representing the chat document
    chat_document = {
        'id':  str(uuid.uuid4()),  
        'prompt': user_prompt,
        'completion': response['choices'][0]['message']['content'],
        'completionTokens': str(response['usage']['completion_tokens']),
        'promptTokens': str(response['usage']['prompt_tokens']),
        'totalTokens': str(response['usage']['total_tokens']),
        'model': response['model'],
         'name': username,
        'vector': prompt_vectors
    }
    # Insert the chat document into the Cosmos DB container
    container.create_item(body=chat_document)
    print("item inserted into cache.", chat_document)

def clearcache ():
   
    client = CosmosClient(HOST, {'masterKey': MASTER_KEY})
    mydbt = client.get_database_client(dbsource)   
  
    
      
# Create the vector embedding policy to specify vector details
    vector_embedding_policy = {
    "vectorEmbeddings": [ 
        { 
            "path":"/vector" ,
             "dataType":"float32",
            "distanceFunction":"cosine",
            "dimensions":1536
        }, 
    ]
}

# Create the vector index policy to specify vector details
    indexing_policy = { 
    "vectorIndexes": [ 
        {
            "path": "/vector", 
            "type": "diskANN"
            
        }
    ]
    } 
   
    mydbt.delete_container(cachecol)


# Create the cache collection with vector index
    try:
        mydbt.create_container_if_not_exists( id=cachecol, 
                                                  partition_key=PartitionKey(path='/id'), 
                                                  indexing_policy=indexing_policy,
                                                  vector_embedding_policy=vector_embedding_policy
                                                ) 
        print('Container with id \'{0}\' created'.format(id)) 

    except exceptions.CosmosHttpResponseError: 
        raise 
    return "Cache cleared."

def generatecompletionede(user_prompt, vector_search_results, chat_history):
    
    system_prompt = '''
    You are an intelligent assistant for yourdata . You are designed to provide helpful answers to user questions about  your data.
    You are friendly, helpful, and informative and can be lighthearted. Be concise in your responses, but still friendly.
        - Only answer questions related to the information provided below. 
        - Write two lines of whitespace between each answer in the list.
    '''

    # Create a list of messages as a payload to send to the OpenAI Completions API

    # system prompt
    
    messages = [{'role': 'system', 'content': system_prompt}]
    
    #chat history
    for chat in chat_history:
        messages.append({'role': 'user', 'content': chat['prompt'] + " " + chat['completion']})
    
    #user prompt
    messages.append({'role': 'user', 'content': user_prompt})

    #vector search results
    for result in vector_search_results:
        messages.append({'role': 'system', 'content': result['text']})

    
    # Create the completion
    response = get_completion(openai_client, openai_chat_model, messages)
    print("Response from openai", response)
    
    return response

def chat_completion(user_input,username):

    # Generate embeddings from the user input
    user_embeddings = generate_embeddings(openai_client, user_input)
    
    # Query the chat history cache first to see if this question has been asked before
    cache_results = cache_search(user_embeddings ,username,0.99, 1)

    if len(cache_results) > 0:
        return cache_results[0]['completion'], True
    else:
        # Perform vector search on the movie collection
        print("\n New result\n")
        search_results = get_similar_docs(openai_client, dbsource, user_input, 5)
        print("\n search result done\n")
        
        
        print("Getting Chat History\n")
        # Chat history
        chat_history = get_chat_history(username,1)

        # Generate the completion
        print("Generating completions \n")
        completions_results = generatecompletionede(user_input, search_results, chat_history)

        print("Caching response \n")
        # Cache the response
        cacheresponse(user_input, user_embeddings, completions_results,username)

        print("\n")
        # Return the generated LLM completion
        return completions_results['choices'][0]['message']['content'], False



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/clearcache', methods=['POST'])
def clear_cache():
    result = clearcache()
    return jsonify(result)

@app.route('/loaddata', methods=['POST'])
def load_data():
    result = loaddata(dbsource,'argus')
    ReadFeed('argus')
    return jsonify(result)

@app.route('/send_message', methods=['POST'])
def send_message():
    if request.method=='POST' : 
        
        username =request.form['username']
        message = request.form['message']

        start_time = time.time()
        response_payload, cached = chat_completion(message,username)
        end_time = time.time()
        elapsed_time = round((end_time - start_time) * 1000, 2)
        response = response_payload
        
        details = f"\n (Time: {elapsed_time}ms)"
        if cached:
            details += " (Cached)"
        chat_history.append([message, response + "for "+ username + details])
      
        return render_template("index.html",message=chat_history   )

@app.route('/clear', methods=['POST'])
def clear():
    global chat_history
    chat_history = []
    return jsonify(chat_history=chat_history)

if __name__ == '__main__':
    app.run(debug=True)