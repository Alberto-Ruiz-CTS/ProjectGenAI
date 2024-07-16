import const
import json
import requests

headers = {"Authorization": f"Bearer {const.API_KEYedu}"}

url = "https://api.edenai.run/v2/text/chat"

customer_email = """
Arrr, I be fuming that me blender lid \
flew off and splattered me kitchen walls \
with smoothie! And to make matters worse,\
the warranty don't cover the cost of \
cleaning up me kitchen. I need yer help \
right now, matey!
"""

style = """American English in a calm and respectful tone"""

payload = {
    "providers": "openai",
    "text": f"""Translate the text that is delimited by triple backticks into a style that is {style}. text: ```{customer_email}```""",
    "chatbot_global_action": "Act as a language expert",
    "previous_history": [],
    "temperature": 0.0,
    "max_tokens": 150,
}

response = requests.post(url, json=payload, headers=headers)

result = json.loads(response.text)
print(result['openai']['generated_text'])

