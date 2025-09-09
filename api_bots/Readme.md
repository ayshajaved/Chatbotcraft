For an OpenAI API chatbot:
1. Copy `.env.example` to `.env` and add your OpenAI API key
Go to openai.com and find our api key

2. Run the OpenAI chatbot: `python api/openai_bot.py`

For Mistral Chatbot:
1. Copy `.env.example` to `.env` and add your mistral API key
Go to mistral.ai and find our api key

2. Run the Mistral chatbot: `python api/mistral_bot.py`


# E-Commerce Chatbot API with Custom Instructions

This project sets up a RESTful API for an e-commerce customer service chatbot using Python, FastAPI, and the Hugging Face Transformers library. The chatbot is based on Microsoft's **DialoGPT-small**, fine-tuned or used as-is with custom instructions to handle queries about products, orders, and returns. The API accepts user inputs via HTTP POST requests and returns chatbot responses.

## Purpose
The chatbot API is designed for e-commerce platforms to automate customer support, answering common questions like:
- "What is the price of the blue shirt?"
- "How do I return an item?"
- "Can you track my order?"

Custom instructions ensure the chatbot provides accurate, context-specific responses while maintaining a friendly tone.

## Prerequisites
- **Python**: Version 3.8 or higher.
- **Hardware**: CPU is sufficient; GPU recommended for faster inference.
- **Operating System**: Windows, macOS, or Linux.
- **Basic Knowledge**: Familiarity with Python, REST APIs, and JSON.

## Required Libraries
| Library | Purpose | Installation Command |
|---------|---------|----------------------|
| `transformers` | Provides pre-trained models (DialoGPT) and tokenizers. | `pip install transformers` |
| `torch` | PyTorch for model inference. | `pip install torch` |
| `fastapi` | Web framework for building the API. | `pip install fastapi` |
| `uvicorn` | ASGI server to run the FastAPI app. | `pip install uvicorn` |
| `spacy` | Text preprocessing (lemmatization, tokenization). | `pip install spacy; python -m spacy download en_core_web_sm` |
| `nltk` | Additional text processing (e.g., tokenization for evaluation). | `pip install nltk` |

Install all dependencies:
```bash
pip install transformers torch fastapi uvicorn spacy nltk
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt')"
```

## Setup Instructions

### Step 1: Prepare the Environment
1. Create a project directory:
   ```bash
   mkdir ecommerce_chatbot_api
   cd ecommerce_chatbot_api
   ```
2. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the required libraries (see above).

### Step 2: Prepare Custom Instructions
Custom instructions guide the chatbot’s behavior. For an e-commerce chatbot, instructions might include:
- Respond concisely and professionally.
- Always include a call-to-action (e.g., "Visit our website for more details").
- Handle unknown queries with: "I’m sorry, I don’t have that information. Please contact support@ecommerce.com."

Store instructions in a configuration file (`config.json`):
```json
{
  "tone": "professional",
  "default_response": "I'm sorry, I don't have that information. Please contact support@ecommerce.com.",
  "call_to_action": "Visit our website for more details."
}
```

### Step 3: Implement the Chatbot API
Create a Python script (`chatbot_api.py`) to load the DialoGPT model, process user inputs with custom instructions, and serve the API using FastAPI.

### Step 4: Test the API
1. Run the API:
   ```bash
   python chatbot_api.py
   ```
2. The API will be available at `http://localhost:8000`.
3. Send a POST request using a tool like `curl`, Postman, or Python `requests`:
   ```bash
   curl -X POST "http://localhost:8000/chat" -H "Content-Type: application/json" -d '{"user_input": "What is the price of the blue shirt?"}'
   ```
   Expected response:
   ```json
   {"response": "The blue shirt costs $29.99. Visit our website for more details."}
   ```

### Step 5: Deploy the API
- **Local Deployment**: Run `uvicorn chatbot_api.py:app --host 0.0.0.0 --port 8000`.
- **Cloud Deployment**: Use platforms like AWS, Heroku, or Google Cloud. For Heroku:
  1. Create a `Procfile`: `web: uvicorn chatbot_api:app --host 0.0.0.0 --port $PORT`
  2. Deploy using `heroku create` and `git push heroku main`.
- **Security**: Add authentication (e.g., OAuth2 with `fastapi.security`) and HTTPS.

### Step 6: Evaluate and Monitor
- **Test Cases**: Try queries like "Track my order," "Return policy," or invalid inputs.
- **Metrics**: Use BLEU score for response quality:
  ```python
  from nltk.translate.bleu_score import sentence_bleu
  reference = "The blue shirt costs $29.99.".split()
  generated = "The blue shirt is priced at $29.99.".split()
  print(sentence_bleu([reference], generated))
  ```
- **Logging**: Log user queries and responses for analysis (e.g., using Python’s `logging` module).

## Code
Below is the implementation of `chatbot_api.py`:

```python
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI
import uvicorn
import spacy
from nltk.tokenize import word_tokenize
import nltk

# Initialize NLTK and spaCy
nltk.download('punkt')
nlp = spacy.load("en_core_web_sm")

# Load custom instructions
with open("config.json", "r") as f:
    config = json.load(f)

# Load model and tokenizer
model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Preprocessing function
def preprocess_input(user_input):
    return ' '.join([token.lemma_.lower() for token in nlp(user_input)])

# Response generation with custom instructions
def generate_response(user_input, max_length=50):
    processed_input = preprocess_input(user_input)
    input_ids = tokenizer.encode(processed_input + tokenizer.eos_token, return_tensors='pt')
    output = model.generate(input_ids, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output[0], skip_special_tokens=True).split('</s>')[-1].strip()
    
    # Apply custom instructions
    if not response:
        response = config["default_response"]
    response += f" {config['call_to_action']}"
    return response

# FastAPI app
app = FastAPI(title="E-Commerce Chatbot API")

@app.post("/chat")
async def chat(user_input: str):
    response = generate_response(user_input)
    return {"response": response}

# Run the app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Usage
1. Save the code as `chatbot_api.py`.
2. Create `config.json` with custom instructions.
3. Run the script: `python chatbot_api.py`.
4. Test with a POST request:
   ```python
   import requests
   response = requests.post("http://localhost:8000/chat", json={"user_input": "How do I return an item?"})
   print(response.json())
   ```

## Customization
- **Tone**: Modify `config.json` to change the tone (e.g., "friendly", "formal").
- **Model**: Replace DialoGPT with T5 or BlenderBot for different use cases.
- **Endpoints**: Add routes for specific tasks (e.g., `/track_order`).
- **Integration**: Connect to Slack or Telegram using their APIs.

## Troubleshooting
- **Model Loading Errors**: Ensure sufficient memory (4GB+ RAM for DialoGPT-small).
- **API Not Responding**: Check port 8000 is free; use `lsof -i :8000` to debug.
- **Poor Responses**: Fine-tune the model on your data (see README 2).

## Future Improvements
- Add authentication for secure access.
- Implement session management for conversational context.
- Use retrieval-augmented generation (RAG) for accurate responses.