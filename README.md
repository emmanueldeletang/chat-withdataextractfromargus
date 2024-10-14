# chat-withdataextractfromargus
how to plug a chat web app connect to argus accelerator to show the data for end user 

# architecture 
![image](https://github.com/user-attachments/assets/a758cced-6521-41f4-93f2-b1827797c918)


## Features
- Vector search using Azure Cosmos DB for NoSQL
- Create embeddings using Azure OpenAI text-embedding
- Use cosmosdb Nosql as cache to save latency

## Requirements
- Tested only with Python 3.12
- Azure OpenAI account
- Azure Cosmos DB for NoSQL account
- Getting data from ARGUS ACCELERATOR : https://github.com/Azure-Samples/ARGUS
-  Automated Retrieval and GPT Understanding System Argus Panoptes, in ancient Greek mythology, was a giant with a hundred eyes and a servant of the goddess Hera. His many eyes made him an excellent watchman, as some of his eyes would always remain open while the others slept, allowing him to be ever-vigilant.


## Setup
- Create virtual environment: python -m venv .venv
- Activate virtual ennvironment: .venv\scripts\activate
- Install required libraries: pip install -r requirements.txt
- Replace keys with your own values in Argus.env
- don't forget to have the model openAI one text-embbeding and one GPT4-o  .. 

## Demo script
- Open "Argusconnect.ipynb" python notebook
- Connect inside the .env to the ARGUS Cosmosdb account , and be sure the features Vector Search for NoSQL API (preview) is turn on , if you turn on , you may wait several minutes to execute the code 
- Run the cells to create create the container and populate the Cosmos DB database with different data 
- The last cell launch Gradio UI 
- if you ingest the samples who are in argus repository  :

you can ask for : Dottore di Ricerca Responsabile Chirurgia Oculistica Clinica Rugani ? 
the ask should be based on your documents.... 
