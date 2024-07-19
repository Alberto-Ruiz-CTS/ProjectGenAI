import gradio as gr
import const as cs
import json
import requests
from pypdf import PdfReader
from utils import word_wrap
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

def clear_history():
    global conversation_history
    conversation_history = []
    return ""

def main():

    embedding_function = SentenceTransformerEmbeddingFunction()
    chroma_collection_trafico = load_chroma(filename='/teamspace/studios/this_studio/ProjectGenAI/data/BOE-020_Codigo_de_Trafico_y_Seguridad_Vial.pdf', 
                                            collection_name='boe_normativa_trafico', embedding_function=embedding_function)

    # Define the Gradio interface
    with gr.Blocks() as demo:
        with gr.Tabs():
            with gr.TabItem("Simple Chatbot"):
                gr.Markdown("## Simple Chatbot")
                chatbot_input = gr.Textbox(label="Enter your message")
                chroma_db = chroma_collection_trafico
                mode = gr.Dropdown(label="Select complexity mode", choices=["Basic", "Expansion", "Multiple"], value="Basic")
                chatbot_output = gr.Textbox(label="Response")
                chatbot_button = gr.Button("Send")
                chatbot_button.click(generate_chatbot_answer, [chatbot_input, chroma_db, user_info_input], chatbot_output)
            
            with gr.TabItem("Receipt Scanner & Chatbot"):
                gr.Markdown("## Receipt Scanner")
                receipt_image = gr.Image(type="file", label="Upload Receipt Image")
                receipt_button = gr.Button("Add")
                receipt_output = gr.Textbox(label="Status")
                receipt_button.click(add_receipt, receipt_image, receipt_output)
                
                gr.Markdown("## Query Receipts Database")
                query_input = gr.Textbox(label="Ask a question about the receipts")
                query_output = gr.Textbox(label="Response")
                query_button = gr.Button("Ask")
                query_button.click(query_receipts, query_input, query_output)

    # Run the app
    demo.launch(share=True)

if __name__ == "__main__":
    main()
