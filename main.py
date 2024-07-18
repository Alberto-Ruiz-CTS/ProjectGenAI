import pandas as pd
from PIL import Image
import io
import easyocr
import pickle
from pydantic import BaseModel, Field
import pickle
from langchain.output_parsers import PydanticOutputParser
import requests
import const
import json

headers = {"Authorization": f"Bearer {const.API_KEYedu}"}

url = "https://api.edenai.run/v2/text/chat"

customer_email = """
Eres estúpido y no sabes redactar bien un simple mail, aprende a escribir!
"""

style = """American English in a calm and respectful tone"""

system_template = """
    You have to perform the task of extracting information from data.
    
    Extract the data following the format:
    {format_instructions}
    
    Example:
    Data: 'Max 5 Cafe 2200 Fifth Ave Seattle  WA   98121 206-441-9785 9999998  Trainee Tbl 31/1 Chk   1441 Gst 2 Jul19' 14 12:36PM BenSPECIAL 12,.50 YELP Open $ Disc 2.05 - Wait 10 % Open % Disc 1,05- Caprese 10.00 Add Chicken 1,00 YELP 
            Open $ Disc 1,80 - Pineapple Bellin 7,00 YELP Open $ Disc 1,15- Subtota] 24,45 Tax 2,32 01:O2PM Total 26 77'
    Output: {{ "name": "Max's Cafe", "address": "2200 Fifth Ave Seattle, WA 98121", "date": "07-19-2014", "total": "$26.77", "category": "Restaurant" }}
    """
 
prompt_template = """ Data: {data} """

class Ticket(BaseModel):
    """TicketDatamodel"""
    name : str = Field(default = None, description = "Name of the establishment where the purchase was made")
    address : str = Field(default = None, description = "Address of the establishment")
    date : str = Field(default = None, description = "Date of the ticket")
    total: str = Field(default = None, description = "Total money amount of the ticket")
    category: str = Field(default = None, description = "Choose one of the following: restaurant, groceries, leisure, gas, other")

def pydanticParser():
    return PydanticOutputParser(pydantic_object = Ticket)

parser = pydanticParser()

def instructionsFormat(parser, textOcr):
   
    system_instructions = system_template.format(
        format_instructions=parser.get_format_instructions(),
    )
 
    prompt = prompt_template.format(
        data=textOcr,
    )
 
    return system_instructions, prompt

l_textOcr = []
l_textOcr.append('SPEEDUAY 0006661 La Porte IN 76350 TRANI : 2387827 5/19/2017 6 : 29 PI 02 #2 Diesel 10.875 @ $2.399/GaL Gas ToTaL $26 09 Tax $0.00 ToTaL $26 09 Speedvay MC Card Num XXXXXXXXXXXX4737 TERM: 0050006661001 TRANS  TYPE: CAPTURE APPR#: 3 0886B EnTRY Method: ICR Speedy Points New 1413 Dalance: 135837 vnw coh Punp Rewards Earned SPEEDVAY')
l_textOcr.append("MASTORIS DINER 144 ROUTE 130 BORDENTOWN , NJ #(609)-298-4650 276 ANGELA POWELL 2/3 1885 GST 2 O4Nov ' 17 9;32AM HAM&EGG BURRITO 9,99 1 SS PANCAKES 5.99 ADD STRWBER 3.00 INSIDE #SPEC PREP* 2 HOT TEA 4,98 Subtota] 23,96 Tax 1.65 10;00 Payment Due 25 61 EAGLES 7-1 COWBOYS 4-3 REDSKINS 3-4 GIANTS 1-6 GO EAGLES !")


# Load the content of l_textOcr from the pickle file
with open('/teamspace/studios/this_studio/ProjectGenAI/data/receipts/l_textOcr.pkl', 'rb') as file:
    l_textOcr = pickle.load(file)

for i in range(2):
    system_instructions, prompt = instructionsFormat(parser, l_textOcr[i])

    payload = {
        "providers": "openai",
        "text":  prompt,
        "chatbot_global_action": system_instructions,
        "previous_history": [],
        "temperature": 0.0,
        "max_tokens": 150,
    }

    response = requests.post(url, json=payload, headers=headers)

    result = json.loads(response.text)
    print(result['openai']['generated_text'])
    print(type(result))

