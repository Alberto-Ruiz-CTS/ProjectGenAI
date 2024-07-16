import gradio as gr
import const
import json
import requests
from pypdf import PdfReader
from helper_utils import word_wrap
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


# Authorization header for the API
headers = {"Authorization": f"Bearer {const.API_KEYedu}"}

# URL for the API
url = "https://api.edenai.run/v2/text/chat"

# Read pdf BOE
reader = PdfReader("/teamspace/studios/this_studio/ProjectGenAI/BOE-A-1978-31229-consolidado.pdf")
pdf_texts = [p.extract_text().strip() for p in reader.pages]

# Filter the empty strings
pdf_texts = [text for text in pdf_texts if text]

# We divide our pdf in chunks in that order, so no chunk is larger than 1000
character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],
    chunk_size=1000,
    chunk_overlap=0
)
character_split_texts = character_splitter.split_text('\n\n'.join(pdf_texts))

# Redivide our chunks according to token count so our chunks capture all the meaning
token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=256)

token_split_texts = []
for text in character_split_texts:
    token_split_texts += token_splitter.split_text(text)

print(f"\nTotal chunks: {len(token_split_texts)}")

# Uses the BERT architecture but slight different architecture to embed whole chunks, not just single words
embedding_function = SentenceTransformerEmbeddingFunction()

# We embed all our chunks and pass them into chroma to build the framework
chroma_client = chromadb.Client()
chroma_collection = chroma_client.create_collection("boe", embedding_function=embedding_function)

ids = [str(i) for i in range(len(token_split_texts))]

chroma_collection.add(ids=ids, documents=token_split_texts)

# Querying RAG, embedding our query and Retrieving similar chunks

query = "¿Puede alguien ser miembro de varias Cámaras?"

results = chroma_collection.query(query_texts=[query], n_results=5)
retrieved_documents = results['documents'][0]

for document in retrieved_documents:
    print(word_wrap(document))
    print('\n')







# Initial conversation history
conversation_history = []

def chatbot_response(user_input):
    global conversation_history

    # Append the user's message to the conversation history
    conversation_history.append({"role": "user", "content": user_input})

    # Construct the payload for the API request
    payload = {
        "providers": "openai",
        "text": user_input,
        "chatbot_global_action": "Act as a language expert",
        "previous_history": [{"role": msg["role"], "message": msg["content"]} for msg in conversation_history],
        "temperature": 0.0,
        "max_tokens": 150,
    }

    # Debug print the payload
    print("Request Payload:", json.dumps(payload, indent=2))

    # Send the request to the API
    response = requests.post(url, json=payload, headers=headers)
    
    # Check for successful response
    if response.status_code != 200:
        return f"Error: Received status code {response.status_code} with response: {response.text}"
    
    # Parse the JSON response
    try:
        result = json.loads(response.text)
        print("API Response:", result)  # Debug print
        bot_response = result['openai']['generated_text']
    except KeyError as e:
        return f"Error: Key {e} not found in the response. Full response: {result}"
    except json.JSONDecodeError:
        return "Error: Failed to parse the response from the API."

    # Append the chatbot's response to the conversation history
    conversation_history.append({"role": "assistant", "content": bot_response})

    # Return the conversation history
    conversation_text = ""
    for message in conversation_history:
        if message["role"] == "user":
            conversation_text += f"User: {message['content']}\n"
        else:
            conversation_text += f"Bot: {message['content']}\n"
    
    return conversation_text

def clear_history():
    global conversation_history
    conversation_history = []
    return ""

# Create the Gradio interface
with gr.Blocks() as demo:
    user_input = gr.Textbox(lines=2, placeholder="Enter your message here...", label="Your Message")
    conversation_output = gr.Textbox(lines=20, label="Conversation History")
    
    submit_button = gr.Button("Send")
    clear_button = gr.Button("Clear History")
    
    submit_button.click(fn=chatbot_response, inputs=user_input, outputs=conversation_output)
    clear_button.click(fn=clear_history, inputs=None, outputs=conversation_output)

# Launch the interface
#demo.launch(share=True)
