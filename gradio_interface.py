import gradio as gr
import const as cs
from functools import partial
import json
import requests
from pypdf import PdfReader
from utils import word_wrap
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from utils import load_chroma_pdf, load_chroma_tickets, chatbot_pdf, chatbot_ticket, process_ticket
from ticket import Ticket

def clear_history():
    global conversation_history
    conversation_history = []
    return ""

def main():

    # Define the Gradio interface
    with gr.Blocks() as demo:
        with gr.Tabs():
            with gr.TabItem("DGT Chatbot"):
                gr.Markdown("## Ask whatever to the DGT chatbot!")
                chatbot_input = gr.Textbox(label="Enter your message")
                mode = gr.Dropdown(label="Select complexity mode", choices=["Basic", "Expansion", "Multiple"], value="Basic")
                chatbot_output = gr.Textbox(label="Response")
                chatbot_button = gr.Button("Send")
                chatbot_button.click(chatbot_pdf, [chatbot_input, mode], chatbot_output)
            
            with gr.TabItem("Receipt Scanner & Chatbot"):
                gr.Markdown("## Receipt Scanner")
                receipt_image = gr.Image(type="filepath", label="Upload Receipt Image")
                receipt_button = gr.Button("Add")
                receipt_output = [
                    gr.Textbox(label="Establishment name"),
                    gr.Textbox(label="Address"),
                    gr.Textbox(label="Purchase Date"),
                    gr.Textbox(label="Category"),
                    gr.Textbox(label="Total")
                ]
                receipt_button.click(process_ticket, [receipt_image], receipt_output)
                
                gr.Markdown("## Query Receipts Database")
                query_input = gr.Textbox(label="Ask a question about the receipts")
                query_output = gr.Textbox(label="Response")
                query_button = gr.Button("Send")
                query_button.click(chatbot_ticket, [query_input], query_output)

    # Run the app
    demo.launch(share=True)

if __name__ == "__main__":
    main()
