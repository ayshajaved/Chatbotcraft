#Importing required Libraries
import os                       #accessing environment variables
from openai import OpenAI       #Dealing with openai key 
from dotenv import load_dotenv  #loads the .env file

class openai_bot:
    def __init__(self):
        load_dotenv() #laoding the .env file variables
        api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI(api_key=api_key) #initializing the client of openai

    def get_openai_response(self):
        pass