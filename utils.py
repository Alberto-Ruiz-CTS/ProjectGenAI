import chromadb
from chromadb.config import Settings

from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
import numpy as np
from pypdf import PdfReader
from tqdm import tqdm
import io
from PIL import Image
import easyocr
import const as cs
import requests
import json
from langchain.output_parsers import PydanticOutputParser
from ticket import Ticket
import pickle


def _read_pdf(filename):
    reader = PdfReader(filename)
    
    pdf_texts = [p.extract_text().strip() for p in reader.pages]

    # Filter the empty strings
    pdf_texts = [text for text in pdf_texts if text]
    return pdf_texts


def _chunk_texts(texts):
    character_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=1000,
        chunk_overlap=0
    )
    character_split_texts = character_splitter.split_text('\n\n'.join(texts))

    token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=256)

    token_split_texts = []
    for text in character_split_texts:
        token_split_texts += token_splitter.split_text(text)

    return token_split_texts


def create_and_load_chroma_tickets(json_tickets, collection_name, embedding_function, persist_path):
    
    # Initialize Chroma client with persistence settings
    chroma_client = chromadb.PersistentClient(path=persist_path)

    # Create or get the collection
    chroma_collection = chroma_client.get_or_create_collection("ticket_collection")

    # Prepare IDs and add documents to the collection
    ids = [str(i) for i in range(len(json_tickets))]
    chroma_collection.add(ids=ids, documents=json_tickets)

    return chroma_collection

def load_chroma_tickets(collection_name, persist_path):
    
    # Initialize Chroma client with persistence settings
    chroma_client = chromadb.PersistentClient(path=persist_path)

    # Create or get the collection
    chroma_collection = chroma_client.get_or_create_collection(collection_name)

    return chroma_collection

def create_and_load_chroma_pdf(filename, collection_name, embedding_function, persist_path):
    
    texts = _read_pdf(filename)
    chunks = _chunk_texts(texts)

    # Initialize Chroma client with persistence settings
    chroma_client = chromadb.PersistentClient(path=persist_path)

    # Create or get the collection
    chroma_collection = chroma_client.get_or_create_collection(name=collection_name, embedding_function=embedding_function)

    # Prepare IDs and add documents to the collection
    ids = [str(i) for i in range(len(chunks))]
    chroma_collection.add(ids=ids, documents=chunks)

    return chroma_collection

def load_chroma_pdf(collection_name, persist_path):

    # Initialize Chroma client with persistence settings
    chroma_client = chromadb.PersistentClient(path=persist_path)
    # Create or get the collection
    chroma_collection = chroma_client.get_or_create_collection(name=collection_name)

    return chroma_collection

def word_wrap(string, n_chars=72):
    # Wrap a string at the next space after n_chars
    if len(string) < n_chars:
        return string
    else:
        return string[:n_chars].rsplit(' ', 1)[0] + '\n' + word_wrap(string[len(string[:n_chars].rsplit(' ', 1)[0])+1:], n_chars)

   
def project_embeddings(embeddings, umap_transform):
    umap_embeddings = np.empty((len(embeddings),2))
    for i, embedding in enumerate(tqdm(embeddings)): 
        umap_embeddings[i] = umap_transform.transform([embedding])
    return umap_embeddings


# OCR Functions

# Create image from its bytes
def imageObjCreation(image_bytes):
    image_file = io.BytesIO(image_bytes)
    return Image.open(image_file)

def readImgOCR(image_bytes):

    reader = easyocr.Reader(['en', 'es', 'fr'])
    textEasyOcr = reader.readtext(image_bytes,detail=0)
    textEasyOcr = " ".join(textEasyOcr)

    return textEasyOcr

def LLMModelCall(system_instructions, prompt):
    
    payload = {
        "providers": "openai",
        "text": prompt,
        "chatbot_global_action": system_instructions,
        "previous_history": [],
        "temperature": 0.0,
        "max_tokens": 150,
    }
 
    response = requests.post(cs.url, json=payload, headers=cs.headers_alberto)
    result = response.json()
    result = response.json()

    return result['openai']['generated_text']

def instructionsFormat(parser, textOcr):
   
    system_instructions = cs.SYSTEM_TEMPLATE.format(
        format_instructions=parser.get_format_instructions(),
    )
 
    prompt = cs.PROMPT_TEMPLATE.format(
        data=textOcr,
    )
 
    return system_instructions, prompt

# Query PDFs: basic, expansion generated and using multiple queries

# Function to expand users query witha hypothetical answer
def augment_generated_query(query):
    
    payload = {
    "providers": "openai",
    "text": query,
    "chatbot_global_action": "Eres un asistente experto en la normativa de tráfico de la Dirección General de Tráfico (DGT), el organismo \
                              un organismo autónomo del Gobierno de España dependiente del Ministerio del Interior responsable de la ejecución de la \
                              política vial en las vías de titularidad estatal de España."
                "Sugiere una respuesta de ejemplo a la pregunta realizada, algo como lo que podría encontrarse en la normativa de la DGT",
    "previous_history": [],
    "temperature": 0.0,
    "max_tokens": 150,
    }

    # Send the request to the API
    response = requests.post(cs.url, json=payload, headers=cs.headers_alberto)
    result = json.loads(response.text)
    augmented_query = result['openai']['generated_text']
    
    return augmented_query

# Function that augments a query with multiple questions related to the original query
def augment_multiple_query(query):
    
    payload = {
    "providers": "openai",
    "text": query,
    "chatbot_global_action": "Eres un asistente experto en la normativa de tráfico de la Dirección General de Tráfico (DGT), el organismo \
                              un organismo autónomo del Gobierno de España dependiente del Ministerio del Interior responsable de la ejecución de la \
                              política vial en las vías de titularidad estatal de España."
                    "Sugiere hasta cinco preguntas adicionales relacionadas para ayudarles a encontrar la información que necesitan para la pregunta proporcionada."
                    "Sugiere solo preguntas cortas sin oraciones compuestas. Sugiere una variedad de preguntas que cubran diferentes aspectos del tema."
                    "Asegúrate de que sean preguntas completas y que estén relacionadas con la pregunta original."
                    "La salida debe ser una pregunta por línea. No numeres las preguntas.",
    "previous_history": [],
    "temperature": 0.0,
    "max_tokens": 150,
    }

    # Send the request to the API
    response = requests.post(cs.url, json=payload, headers=cs.headers_alberto)
    result = json.loads(response.text)
    augmented_queries = result['openai']['generated_text']
    augmented_queries = augmented_queries.split('\n')
    
    return augmented_queries

# Function that finally calls the LLM with the documents retrieved 
def chatbot_pdf(original_query,mode='Basic'):

    chroma_collection = load_chroma_pdf(collection_name='boe_normativa_trafico', 
                                        persist_path='/teamspace/studios/this_studio/ProjectGenAI/data/chroma_databases')
    
    retrieved_documents = retrieve_documents(original_query,chroma_collection,mode)
    information = "\n\n".join(retrieved_documents)
    final_query = f"Pregunta: {original_query} \n Información: {information}"

    payload = {
    "providers": "openai",
    "text": final_query,
    "chatbot_global_action": "Eres un asistente experto en la normativa de tráfico de la Dirección General de Tráfico (DGT), el organismo \
                              un organismo autónomo del Gobierno de España dependiente del Ministerio del Interior responsable de la ejecución de la \
                              política vial en las vías de titularidad estatal de España."
            "Tu tarea es ayudar a los usuarios a comprender y responder preguntas sobre el contenido de la normativa de tráfico de la DGT. "
            "Te proporcionaré la pregunta del usuario y la información relevante extraída de la normativa. "
            "Responde a la pregunta del usuario utilizando ÚNICAMENTE la información proporcionada. ",
    "previous_history": [],
    "temperature": 0.0,
    "max_tokens": 150,
    }

    # Send the request to the API
    response = requests.post(cs.url, json=payload, headers=cs.headers_alberto)
    result = json.loads(response.text)
    final_response = result['openai']['generated_text']

    return final_response

def format_question(query):

    # Format question with LLM call
    instructions = f'''Ejemplos:
                     
                     Pregunta: Cúantas compras se realizaron en el trader joes el 4 de julio de 2023? 
                     Respuesta: name: Trader Joe\'s date: 07-04-2023 
                                                                      
                     Pregunta: Cual es la dirección del max cafe donde se hizo una compra de 26 dolares?
                     Respuesta: name: Max Cafe total: 26 
                     
                     A continuación se indica la pregunta que debes contestar:
                     {query}'''

    payload = {
    "providers": "openai",
    "text": instructions,
    "chatbot_global_action": "Eres un asistente encargado de transformar la pregunta del usuario en un formato específico. \
                              El usuario preguntará sobre información de tickets de compra que pueden tener los siguientes campos: name, address, date, category y total. \
                              Primero, identifica los campos mencionados en la pregunta del usuario. Luego, formatea la respuesta como 'campo: valor', donde los campos son \
                              los mencionados anteriormente y el valor es el que pregunta el usuario. \
                              No incluyas información adicional. Solo indica los campos que aparecen en la pregunta e ignora los demás. La fecha debe estar en formato MM-DD-YYYY",
    "previous_history": [],
    "temperature": 0.0,
    "max_tokens": 150,
    }

    # Send the request to the API
    response = requests.post(cs.url, json=payload, headers=cs.headers_alberto)
    result = json.loads(response.text)
    formatted_query = result['openai']['generated_text']

    return formatted_query


def chatbot_ticket(original_query):

    chroma_collection = load_chroma_tickets(collection_name='ticket_collection', 
                                            persist_path='/teamspace/studios/this_studio/ProjectGenAI/data/chroma_databases/tickets_collection')

    formatted_query = format_question(original_query)

    # Retrieve documents based on the chatbot answer
    results = chroma_collection.query(query_texts=[formatted_query], n_results=100)
    retrieved_documents = results['documents'][0]

    information = "\n\n".join(retrieved_documents)
    final_query = f"Pregunta: {original_query} \n Información: {information}"

    payload = {
    "providers": "openai",
    "text": final_query,
    "chatbot_global_action": "Eres un asistente encargado de responder preguntas sobre tickets de compra. Te indicaré la pregunta \
                              hecha por el usuario junto con la información de los tickets necesaria para responder. Usa únicamente dicha información.",
    "previous_history": [],
    "temperature": 0.0,
    "max_tokens": 150,
    }

    # Send the request to the API
    response = requests.post(cs.url, json=payload, headers=cs.headers_alberto)
    result = json.loads(response.text)
    final_response = result['openai']['generated_text']
    

    return final_response



    

def retrieve_documents(original_query,chroma_collection,mode='Basic'):
    
    if mode == 'Multiple':
        
        augmented_queries = augment_multiple_query(original_query)
        queries = [original_query] + augmented_queries
        results = chroma_collection.query(query_texts=queries, n_results=5, include=['documents', 'embeddings'])
        retrieved_documents = results['documents']

        # Deduplicate the retrieved documents
        unique_documents = set()
        for documents in retrieved_documents:
            for document in documents:
                unique_documents.add(document)

        # Final documents passed to the chatbot function
        retrieved_documents = unique_documents
    
    elif mode == 'Expansion':
        
        hypothetical_answer = augment_generated_query(original_query)
        joint_query = f"{original_query} {hypothetical_answer}"
        results = chroma_collection.query(query_texts=joint_query, n_results=5, include=['documents', 'embeddings'])
        retrieved_documents = results['documents'][0]
    
    else:
        
        results = chroma_collection.query(query_texts=[original_query], n_results=5)
        retrieved_documents = results['documents'][0]
    
    return retrieved_documents

# Add ticket and show scanned results
def process_ticket(image_path):
    
    # Process the image to extract text using OCR
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    # image_obj = imageObjCreation(image_bytes)
    ocr_text = readImgOCR(image_bytes)

    # Setup Pydantic parser and format instructions
    parser = PydanticOutputParser(pydantic_object = Ticket)
    system_instructions, prompt = instructionsFormat(parser, ocr_text)

    # Call the LLM model to extract ticket data
    generated_answer = LLMModelCall(prompt, system_instructions)
    ticket_data = parser.parse(generated_answer)
   
    name = ticket_data.name
    address = ticket_data.address
    date = ticket_data.date
    category = ticket_data.category
    total = ticket_data.total

    return name,address,date,category,total