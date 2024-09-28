import os
import requests
import streamlit as st
import numpy as np
import openai
import re
import json
import boto3
import datetime
from dotenv import load_dotenv, find_dotenv
from PyPDF2 import PdfReader
from docx import Document
from css import css
from js import js
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from sklearn.metrics.pairwise import cosine_similarity
from serpapi import GoogleSearch
import tiktoken
import logging
import sagemaker
from sagemaker.huggingface.model import HuggingFaceModel
import atexit

CACHE_EXPIRY_TIME = 60 * 10  # Cache expiry time in seconds (e.g., 10 minutes for weather data)

# Logs
logging.basicConfig(
    filename='chatbot.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables
LAST_ENDPOINT_USAGE = None
ENDPOINT_IDLE_TIMEOUT = 30 * 60  # 30 minutes

def load_api_keys():
    _ = load_dotenv(find_dotenv())
    return {
        "openai_api_key": os.environ.get("OPENAI_API_KEY"),
        "serpapi_api_key": os.environ.get("SERP_API_KEY"),
        "weather_api_key": os.environ.get("WEATHER_API_KEY"),
        "aws_access_key": os.environ.get("AWS_ACCESS_KEY"),
        "aws_secret_access_key": os.environ.get("AWS_SECRET_ACCESS_KEY"),
        "aws_bucket": os.environ.get("AWS_BUCKET"),
        "aws_region": os.environ.get("AWS_REGION"),
        "sentiment_endpoint_name": os.environ.get("SENTIMENT_ENDPOINT_NAME"),
        "sagemaker_role_arn": os.environ.get("SAGEMAKER_ROLE_ARN")
    }

def initialize_openai(api_key):
    openai.api_key = api_key

def initialize_sagemaker_client(aws_region, aws_access_key, aws_secret_access_key):
    return boto3.client(
        'sagemaker-runtime',
        region_name=aws_region,
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_access_key
    )

def check_and_deploy_sagemaker_endpoint(api_keys):
    sagemaker_client = boto3.client(
        'sagemaker',
        region_name=api_keys["aws_region"],
        aws_access_key_id=api_keys["aws_access_key"],
        aws_secret_access_key=api_keys["aws_secret_access_key"]
    )
    endpoint_name = api_keys["sentiment_endpoint_name"]

    # Check if the endpoint exists
    try:
        response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        if response['EndpointStatus'] == 'InService':
            logger.info(f"Endpoint '{endpoint_name}' is already in service.")
            return
        else:
            logger.info(f"Endpoint '{endpoint_name}' exists but is not in service. Status: {response['EndpointStatus']}")
    except sagemaker_client.exceptions.ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'ValidationException':
            logger.info(f"Endpoint '{endpoint_name}' does not exist. Proceeding to deploy.")
            deploy_sagemaker_endpoint(api_keys)
        else:
            logger.error(f"Error checking endpoint status: {e}")
            raise

def deploy_sagemaker_endpoint(api_keys):
    try:
        aws_region = api_keys["aws_region"]
        if not aws_region:
            logger.error("AWS_REGION is not set.")
            raise ValueError("AWS_REGION is not set.")
        role = api_keys["sagemaker_role_arn"]
        endpoint_name = api_keys["sentiment_endpoint_name"]

        # Initialize SageMaker session
        sess = sagemaker.Session(boto_session=boto3.Session(
            region_name=aws_region,
            aws_access_key_id=api_keys["aws_access_key"],
            aws_secret_access_key=api_keys["aws_secret_access_key"]
        ))

        # Hugging Face model configuration
        hub = {
            'HF_MODEL_ID': 'distilbert-base-uncased-finetuned-sst-2-english',
            'HF_TASK': 'text-classification'
        }

        # Create Hugging Face Model
        huggingface_model = HuggingFaceModel(
            env=hub,
            role=role,
            transformers_version='4.17',
            pytorch_version='1.10',
            py_version='py38',
        )

        # Deploy the model to SageMaker
        predictor = huggingface_model.deploy(
            initial_instance_count=1,
            instance_type='ml.m5.xlarge',
            endpoint_name=endpoint_name,
        )
        logger.info(f"Successfully deployed endpoint '{endpoint_name}'.")
    except Exception as e:
        logger.error(f"Failed to deploy endpoint: {e}")
        raise

def delete_sagemaker_endpoint(api_keys):
    try:
        sagemaker_client = boto3.client(
            'sagemaker',
            region_name=api_keys["aws_region"],
            aws_access_key_id=api_keys["aws_access_key"],
            aws_secret_access_key=api_keys["aws_secret_access_key"]
        )
        endpoint_name = api_keys["sentiment_endpoint_name"]
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
        logger.info(f"Successfully deleted endpoint '{endpoint_name}'.")
    except Exception as e:
        logger.error(f"Failed to delete endpoint: {e}")

def save_file_to_s3(file, aws_keys):
    s3 = boto3.client(
        's3',
        aws_access_key_id=aws_keys["aws_access_key"],
        aws_secret_access_key=aws_keys["aws_secret_access_key"],
        region_name=aws_keys["aws_region"]
    )
    file.seek(0)
    s3.upload_fileobj(file, aws_keys["aws_bucket"], file.name)
    return file.name  # Returns the key of the uploaded file

def process_uploaded_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        pdf_reader = PdfReader(uploaded_file)
        text = "".join(page.extract_text() for page in pdf_reader.pages)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(uploaded_file)
        text = "\n".join([para.text for para in doc.paragraphs])
    else:
        text = uploaded_file.getvalue().decode("utf-8")
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter()
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    if not text_chunks:
        raise ValueError("No text chunks to create vector store")
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store

def get_relevant_chunks(user_input, vector_store):
    retriever = vector_store.as_retriever()
    results = retriever.get_relevant_documents(user_input)
    return [result.page_content for result in results]

def get_embedding(text):
    response = openai.embeddings.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    embedding = response['data'][0]['embedding']
    return embedding

def count_tokens(text, model="gpt-3.5-turbo"):
    tokenizer = tiktoken.encoding_for_model(model)
    tokens = tokenizer.encode(text)
    return len(tokens)

def truncate_text(text, max_tokens):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return tokenizer.decode(tokens)

def analyze_sentiment(text, sagemaker_runtime, endpoint_name):
    payload = {
        "inputs": text
    }
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=json.dumps(payload)
    )
    result = json.loads(response['Body'].read().decode())
    label = result[0]['label']
    score = result[0]['score']
    sentiment_score = score if label == 'POSITIVE' else -score
    return sentiment_score  # Returns a value between -1.0 and 1.0

def get_response(user_input, doc_content, chat_history, similar_past_interactions, sagemaker_runtime, sentiment_endpoint_name):
    # Including last N messages for context
    context_messages = [{"role": role, "content": content} for role, content in chat_history[-5:]]
    user_message = {"role": "user", "content": user_input}

    user_sentiment = analyze_sentiment(user_input, sagemaker_runtime, sentiment_endpoint_name)
    if user_sentiment < -0.3:
        system_prompt = "You are a compassionate assistant. Provide a thoughtful and empathetic response to the user's concerns."
    else:
        system_prompt = "You are a helpful assistant. Provide a thoughtful response to the user's concerns."

    # Prepare past interactions context
    past_interactions_text = "\n".join(
        [f"User: {q}\nAssistant: {r}" for q, r in similar_past_interactions]
    )
    system_message = {
        "role": "system",
        "content": f"{system_prompt}\n\nHere are some past interactions that might help:\n{past_interactions_text}"
    }

    total_allowed_tokens = 4096
    if doc_content:
        vector_store = st.session_state.vector_store
        relevant_chunks = get_relevant_chunks(user_input, vector_store)
        context = "\n".join(relevant_chunks)
        prompt_tokens = count_tokens("\n".join([msg["content"] for msg in [system_message] + context_messages + [user_message]]))
        available_tokens = total_allowed_tokens - prompt_tokens - 500
        available_tokens = max(0, available_tokens)
        truncated_context = truncate_text(context, available_tokens)
        context_message = {"role": "system", "content": f"Context:\n{truncated_context}"}
        messages = [system_message, context_message] + context_messages + [user_message]
    else:
        messages = [system_message] + context_messages + [user_message]

    prompt_tokens = count_tokens("\n".join([msg["content"] for msg in messages]))
    max_response_tokens = total_allowed_tokens - prompt_tokens
    max_response_tokens = min(max_response_tokens, 1024)
    max_response_tokens = max(max_response_tokens, 150)

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=max_response_tokens
    )
    return response.choices[0].message.content.strip()

def search_web(query, serpapi_api_key):
    search = GoogleSearch({"q": query, "api_key": serpapi_api_key})
    results = search.get_dict()
    return results.get("organic_results", [])

def get_weather(location, weather_api_key):
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={location['latitude']}&lon={location['longitude']}&appid={weather_api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        weather_description = data['weather'][0]['description']
        temperature = data['main']['temp']
        return f"The weather in your location is {weather_description} with a temperature of {temperature}°C."
    else:
        return "Sorry, I couldn't retrieve the weather information."

def generate_session_name(chat_history):
    """
    Generates a summary-based name for the chat session.
    """
    user_messages = [content for role, content in chat_history if role == 'user']
    conversation_text = "\n".join(user_messages)
    conversation_text = truncate_text(conversation_text, max_tokens=1000)

    system_message = {
        "role": "system",
        "content": "You are an assistant that generates concise and descriptive session titles based on the conversation."
    }
    user_message = {
        "role": "user",
        "content": f"Summarize the main topic of the following conversation in a few words suitable as a session title:\n\n{conversation_text}\n\nSession Title:"
    }

    messages = [system_message, user_message]

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=10,
        temperature=0.5,
        n=1,
        stop=["\n"]
    )
    session_name = response.choices[0].message.content.strip()
    session_name = re.sub(r'[^\w\s\-]', '', session_name)
    return session_name

def add_css_and_html():
    st.markdown(css, unsafe_allow_html=True)

def get_location():
    query_params = st.experimental_get_query_params()
    location = query_params.get("location")
    if location:
        latitude, longitude = map(float, location[0].split(","))
        return {"latitude": latitude, "longitude": longitude}
    return None

def main():
    st.set_page_config(layout="wide")
    st.sidebar.title("Chat Sessions")

    # Load API keys
    api_keys = load_api_keys()

    # Initialize OpenAI
    if api_keys["openai_api_key"] is None:
        st.error("OpenAI API key is not set. Please set it in the .env file.")
        return

    initialize_openai(api_keys["openai_api_key"])

    # Check and deploy SageMaker endpoint if needed
    try:
        check_and_deploy_sagemaker_endpoint(api_keys)
    except Exception as e:
        st.error("Failed to deploy or access the SageMaker endpoint. Please check the logs for more details.")
        logger.error(f"Application failed to start due to SageMaker endpoint issues: {e}")
        return
    
    # Initialize SageMaker client
    sagemaker_runtime = initialize_sagemaker_client(
        api_keys["aws_region"],
        api_keys["aws_access_key"],
        api_keys["aws_secret_access_key"]
    )
    sentiment_endpoint_name = api_keys["sentiment_endpoint_name"]

    # Initialize sessions in session state
    if 'sessions' not in st.session_state:
        st.session_state.sessions = {}
    if 'selected_session' not in st.session_state:
        st.session_state.selected_session = None

    # Display the sidebar menu with chat sessions
    selected_session = st.session_state.selected_session
    for session_id, session_data in st.session_state.sessions.items():
        session_name = session_data['name']
        label = session_name if session_name else f"Chat {session_id[-6:]}"
        is_selected = session_id == selected_session
        with st.sidebar.expander(label, expanded=is_selected):
            if st.button("Open", key=f"open_{session_id}"):
                st.session_state.selected_session = session_id
                st.session_state.chat_history = session_data['chat_history']
                st.session_state.doc_content = session_data.get('doc_content', "")
                st.session_state.vector_store = session_data.get('vector_store', None)
                st.experimental_rerun()
            if st.button("Rename", key=f"rename_{session_id}"):
                new_name = st.text_input("New name", key=f"new_name_{session_id}")
                if new_name:
                    st.session_state.sessions[session_id]['name'] = new_name
                    st.experimental_rerun()
            if st.button("Delete", key=f"delete_{session_id}"):
                del st.session_state.sessions[session_id]
                st.session_state.selected_session = None
                st.session_state.chat_history = []
                st.experimental_rerun()

    # Button to create a new session
    if st.sidebar.button("New Chat"):
        session_id = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        st.session_state.sessions[session_id] = {
            'name': f"Chat {session_id[-6:]}",
            'chat_history': [],
            'doc_content': "",
            'vector_store': None,
            'memory': []
        }
        st.session_state.selected_session = session_id
        st.session_state.chat_history = []
        st.session_state.doc_content = ""
        st.session_state.vector_store = None
        st.session_state.memory = []
        st.experimental_rerun()

    # Load chat history and document content for the selected session
    if st.session_state.selected_session:
        session_data = st.session_state.sessions[st.session_state.selected_session]
        st.session_state.chat_history = session_data.get('chat_history', [])
        st.session_state.doc_content = session_data.get('doc_content', "")
        st.session_state.vector_store = session_data.get('vector_store', None)
        st.session_state.memory = session_data.get('memory', [])

    add_css_and_html()

    # Get user location
    location = get_location()
    if not location:
        st.markdown(js, unsafe_allow_html=True)

    # Create a header container
    header_container = st.container()
    with header_container:
        st.title("Personal Assistant Chatbot")

    # Container for the conversation
    conversation_container = st.container()

    # Form for user input
    input_container = st.container()
    with input_container:
        with st.form(key='chat_form', clear_on_submit=True):
            user_input = st.text_input("Ask your personal assistant anything:")
            uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf", "docx"])
            submit_button = st.form_submit_button(label='Ask')

    if submit_button and user_input:
        current_time = datetime.datetime.now()

        # Add the user's message to chat history immediately
        st.session_state.chat_history.append(("user", user_input))

        # Process uploaded file
        if uploaded_file:
            # Save the file to S3
            file_key = save_file_to_s3(uploaded_file, api_keys)
            # Read file content
            file_content = process_uploaded_file(uploaded_file)
            st.session_state.doc_content += file_content
            # Create vector store and store in session state
            text_chunks = get_text_chunks(st.session_state.doc_content)
            vector_store = get_vector_store(text_chunks)
            st.session_state.vector_store = vector_store

        # Compute embedding for the new question
        try:
            new_question_embedding = np.array(get_embedding(user_input))
        except Exception as e:
            logger.error(f"Error computing embedding: {e}")
            new_question_embedding = np.zeros(1536)

        # Retrieve similar past interactions
        similar_past_interactions = []
        for memory_entry in st.session_state.memory:
            stored_embedding = memory_entry['embedding']
            sim = cosine_similarity([new_question_embedding], [stored_embedding])[0][0]
            if sim > 0.7:  # threshold
                similar_past_interactions.append((memory_entry['question'], memory_entry['response']))

        if "weather" in user_input.lower() and location:
            response = get_weather(location, api_keys["weather_api_key"])
        elif "search" in user_input.lower():
            search_results = search_web(user_input, api_keys["serpapi_api_key"])
            response = "\n".join([f"**{result['title']}**\n{result['link']}\n{result['snippet']}\n" for result in search_results])
        else:
            # Generate response using RAG
            response = get_response(
                user_input,
                st.session_state.doc_content if st.session_state.doc_content else "",
                st.session_state.chat_history,
                similar_past_interactions,
                sagemaker_runtime,
                sentiment_endpoint_name
            )

        # Save the question, response, and embedding in the session memory
        st.session_state.memory.append({
            'question': user_input,
            'response': response,
            'embedding': new_question_embedding,
            'timestamp': current_time.isoformat()
        })

        # Update chat history with the assistant's response
        st.session_state.chat_history.append(("assistant", response))

        # Update session data
        session_data = st.session_state.sessions[st.session_state.selected_session]
        session_data['chat_history'] = st.session_state.chat_history
        session_data['doc_content'] = st.session_state.doc_content
        session_data['vector_store'] = st.session_state.vector_store
        session_data['memory'] = st.session_state.memory

        # Generate a new session name if needed
        if session_data['name'].startswith("Chat "):
            session_name = generate_session_name(st.session_state.chat_history)
            if session_name:
                session_data['name'] = session_name
                st.experimental_rerun()

    # Display chat history when a session is selected
    if st.session_state.selected_session and st.session_state.chat_history:
        with conversation_container:
            for role, content in st.session_state.chat_history:
                role_class = 'user' if role == 'user' else 'assistant'
                icon_url = "https://img.icons8.com/ios-filled/50/000000/user-male-circle.png" if role == 'user' else "https://img.icons8.com/fluency-systems-filled/48/bot.png"
                html = f'''
                    <div class="bubble-container">
                        <div class="bubble {role_class}">{content}</div>
                        <img src="{icon_url}" class="icon {role_class}" />
                    </div>
                    '''
                st.markdown(html, unsafe_allow_html=True)
    
    # Deleting Sagemaker resources to minimize costs            
    def on_exit():
        logger.info("Application is exiting. Deleting SageMaker endpoint.")
        delete_sagemaker_endpoint(api_keys)
    atexit.register(on_exit)

if __name__ == "__main__":
    main()