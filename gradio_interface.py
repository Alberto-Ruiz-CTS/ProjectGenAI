import gradio as gr
import const
import json
import requests

# Authorization header for the API
headers = {"Authorization": f"Bearer {const.API_KEYedu}"}

# URL for the API
url = "https://api.edenai.run/v2/text/chat"

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
demo.launch(share=True)
