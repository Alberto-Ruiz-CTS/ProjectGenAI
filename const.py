# General constants for EdenAI API
url = "https://api.edenai.run/v2/text/chat"
API_KEYedu = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiZGMxMjk2OTQtYjYzYi00ZDA2LWI1YjMtN2QzNDQ2ZDA5YmJmIiwidHlwZSI6ImFwaV90b2tlbiJ9.nisXyVWAaYRT8k0yy8vwQplQCfurL98hNCPrlncW3vo'
API_KEYalberto = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiYzJhMmQ3NWEtYzk1ZS00ZTJhLThhMzQtYThjMzk2MDBlNTc1IiwidHlwZSI6ImFwaV90b2tlbiJ9.7nFBnPVDea_yAu2hA1B85NMG_znd-QJBhvtLP45MdsU'
headers_alberto = {"Authorization": f"Bearer {API_KEYalberto}"}
headers_edu = {"Authorization": f"Bearer {API_KEYedu}"}

# Templates for the OCR
SYSTEM_TEMPLATE = """
    You have to perform the task of extracting information from data.
    
    Extract the data following the format:
    {format_instructions}
    
    Example:
    Data: 'Max 5 Cafe 2200 Fifth Ave Seattle  WA   98121 206-441-9785 9999998  Trainee Tbl 31/1 Chk   1441 Gst 2 Jul19' 14 12:36PM BenSPECIAL 12,.50 YELP Open $ Disc 2.05 - Wait 10 % Open % Disc 1,05- Caprese 10.00 Add Chicken 1,00 YELP 
            Open $ Disc 1,80 - Pineapple Bellin 7,00 YELP Open $ Disc 1,15- Subtota] 24,45 Tax 2,32 01:O2PM Total 26 77'
    Output: {{ "name": "Max's Cafe", "address": "2200 Fifth Ave Seattle, WA 98121", "date": "07-19-2014", "total": "$26.77", "category": "Restaurant" }}
    """

PROMPT_TEMPLATE = """ Data: {data} """