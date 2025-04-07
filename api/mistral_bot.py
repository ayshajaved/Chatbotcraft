#Importing required Libraries
import os                       #accessing environment variables
from mistralai import Mistral   #Dealing with mistral key 
from dotenv import load_dotenv  #loads the .env file

class MistralBot:
    def __init__(self):
        load_dotenv() #laoding the .env file variables
        api_key = os.getenv("Mistral_API_KEY")
        print(api_key) #for debugging
        if not api_key:
            raise ValueError("Mistral_API_KEY not found in environment variables.")
        self.client = Mistral(api_key=api_key) #initializing the client of mistral

    def get_mistral_response(self, user_input):
        #response
        try:
            response = self.client.chat.complete(
                model= "mistral-small",
                messages= [
                    {"role":"system", "content" : "You are a helpful assistant." },
                    {"role": "user", "content": user_input}
                ],
                max_tokens=150,
                temperature=0.7 #0 = strict, 1 = creative
           )
            return response.choices[0].message.content.strip() 
        except Exception as e:
            print("Error: ", e)
        
    def chatbot(self):
        #main loop
        print("***************************")
        print("Welcome to Mistral Chat Bot")
        print("***************************")
        while True:
            user_input = input("YOU: ")
            if user_input.lower() == "quit":
                print("Exiting! Good Bye..")
                break
            response = self.get_mistral_response(user_input)
            print("Mistral Bot: ", response)

#Mian class with static method
class Main:
    @staticmethod
    def run():
        try:
            bot = MistralBot()
            bot.chatbot()
        except Exception as e:
            print("hello")
            print("Error: ",e)
if __name__ == "__main__":
    Main.run()