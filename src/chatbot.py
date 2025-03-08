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
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from sklearn.metrics.pairwise import cosine_similarity
from serpapi import GoogleSearch
import tiktoken
import logging
from css import css
from js import js
from logging.handlers import TimedRotatingFileHandler
import sagemaker
from sagemaker.huggingface.model import HuggingFaceModel
import atexit
import threading
import time
import tempfile
import bcrypt

# -----------------------------------------------------------------------------------
# 1. Set Page Configuration
st.set_page_config(page_title="Personal Assistant Chatbot", page_icon="🤖", layout="wide")

# -----------------------------------------------------------------------------------
# 2. Logger Setup
class LoggerSetup:
    def __init__(self, log_dir="logs", daily_backup_count=7, weekly_backup_count=4, monthly_backup_count=12):
        self.log_dir = log_dir
        self.daily_backup_count = daily_backup_count
        self.weekly_backup_count = weekly_backup_count
        self.monthly_backup_count = monthly_backup_count

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.setup_handlers()

    def setup_handlers(self):
        # Daily log handler
        daily_log_path = os.path.join(self.log_dir, "chatbot_daily.log")
        daily_handler = TimedRotatingFileHandler(
            daily_log_path, when="midnight", interval=1, backupCount=self.daily_backup_count
        )
        daily_handler.setLevel(logging.INFO)
        daily_handler.setFormatter(self.get_formatter())
        self.logger.addHandler(daily_handler)

        # Weekly log handler
        weekly_log_path = os.path.join(self.log_dir, "chatbot_weekly.log")
        weekly_handler = TimedRotatingFileHandler(
            weekly_log_path, when="W0", interval=1, backupCount=self.weekly_backup_count
        )
        weekly_handler.setLevel(logging.INFO)
        weekly_handler.setFormatter(self.get_formatter())
        self.logger.addHandler(weekly_handler)

        # Monthly log handler
        monthly_log_path = os.path.join(self.log_dir, "chatbot_monthly.log")
        monthly_handler = TimedRotatingFileHandler(
            monthly_log_path, when="midnight", interval=30, backupCount=self.monthly_backup_count
        )
        monthly_handler.setLevel(logging.INFO)
        monthly_handler.setFormatter(self.get_formatter())
        self.logger.addHandler(monthly_handler)
        
    def get_formatter(self):
        return logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s')

    def get_logger(self):
        return self.logger

# Initialize the logger
logger_setup = LoggerSetup()
logger = logger_setup.get_logger()

# -----------------------------------------------------------------------------------
# 3. Global Variables
LAST_ENDPOINT_USAGE = None
ENDPOINT_IDLE_TIMEOUT = 30 * 60  # 30 minutes

# -----------------------------------------------------------------------------------
# 4. Utility Functions
def get_secrets(secret_name, region):
    client = boto3.client("secretsmanager", region_name=region)
    
    try:
        response = client.get_secret_value(SecretId=secret_name)
        if "SecretString" in response:
            secret = response["SecretString"]
            return json.loads(secret)
        else:
            secret = response["SecretBinary"]
            return json.loads(secret.decode("utf-8"))
    except client.exceptions.ResourceNotFoundException:
        print(f"Secret {secret_name} not found.")
    except Exception as e:
        print(f"An error occured: {e}")
        
    return None

def load_secrets(secret_name = 'my-chatbot-secrets', region = 'us-east-1'):
    secret_name = 'my-chatbot-secrets'
    region = 'us-east-1'
    secrets = get_secrets(secret_name, region)
    try:
        if secrets:
            return {
                "openai_api_key": secrets.get('openai_api_key'),
                "serpapi_api_key": secrets.get('serpapi_api_key'),
                "weather_api_key": secrets.get('weather_api_key'),
                "aws_access_key": secrets.get('aws_access_key'),
                "aws_secret_access_key": secrets.get('aws_secret_access_key'),
                "aws_region": region,
                "sentiment_endpoint_name": secrets.get('sentiment_endpoint_name'),
                "sagemaker_role_arn": secrets.get('sagemaker_role_arn'),
                "aws_bucket": secrets.get('aws_bucket')
            }
        else:
            raise Exception('Secrets retrieval unsuccessful')
    except Exception as e:
        print(f"An error occured: {e}")

def initialize_openai(api_key):
    openai.api_key = api_key

def get_dynamodb_resource():
    return boto3.resource('dynamodb')  # Credentials are handled via environment variables or IAM roles

def initialize_sagemaker_client(aws_region, aws_access_key, aws_secret_access_key):
    return boto3.client(
        'sagemaker-runtime',
        region_name=aws_region,
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_access_key
    )

def create_endpoint(api_keys):
    sagemaker_client = boto3.client(
        'sagemaker',
        region_name=api_keys["aws_region"],
        aws_access_key_id=api_keys["aws_access_key"],
        aws_secret_access_key=api_keys["aws_secret_access_key"]
    )
    endpoint_name = api_keys["sentiment_endpoint_name"]

    try:
        # Create the new endpoint
        response = sagemaker_client.create_endpoint(
            EndpointName=endpoint_name, 
            EndpointConfigName=endpoint_name
        )
        
        if 'EndpointArn' in response:
            logger.info(f"Endpoint '{endpoint_name}' creation initiated.")
            return True
        else:
            logger.error(f"Endpoint creation response returned without an EndpointArn. Something went wrong.")
            return False

    except sagemaker_client.exceptions.ClientError as e:
        logger.error(f"Error creating the SageMaker endpoint '{endpoint_name}': {e}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during endpoint creation: {e}")
        return False

def describe_sagemaker_endpoint(api_keys):
    try:
        aws_region = api_keys["aws_region"]
        role = api_keys["sagemaker_role_arn"]
        endpoint_name = api_keys["sentiment_endpoint_name"]

        sess = boto3.Session(
            region_name=aws_region,
            aws_access_key_id=api_keys["aws_access_key"],
            aws_secret_access_key=api_keys["aws_secret_access_key"]
        )
        sagemaker_client = sess.client('sagemaker')

        # Check if endpoint configuration already exists
        existing_configs = sagemaker_client.list_endpoint_configs()['EndpointConfigs']
        if any(config['EndpointConfigName'] == endpoint_name for config in existing_configs):
            logger.info(f"Endpoint configuration '{endpoint_name}' already exists. Reusing it.")
            # No action needed if config exists
        else:
            logger.info(f"Creating a new endpoint configuration '{endpoint_name}'.")
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

            # Deploy the model to SageMaker (creates EndpointConfig)
            huggingface_model.deploy(
                initial_instance_count=1,
                instance_type='ml.m5.xlarge',
                 endpoint_name=endpoint_name,
                wait=True,
                endpoint_config_name=endpoint_name  # Reuse the existing name
            )
            logger.info(f"Successfully deployed endpoint configuration '{endpoint_name}'.")

    except Exception as e:
        logger.error(f"Failed to deploy endpoint configuration: {e}")
        raise

def check_and_deploy_sagemaker_endpoint(api_keys):
    sagemaker_client = boto3.client(
        'sagemaker',
        region_name=api_keys["aws_region"],
        aws_access_key_id=api_keys["aws_access_key"],
        aws_secret_access_key=api_keys["aws_secret_access_key"]
    )
    endpoint_name = api_keys["sentiment_endpoint_name"]

    try:
        existing_endpoint = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        status = existing_endpoint['EndpointStatus']
        if status == 'InService':
            logger.info(f"Endpoint '{endpoint_name}' is already InService.")
            st.session_state['endpoint_deployed'] = True
            return
        elif status in ['Creating', 'Updating']:
            logger.info(f"Endpoint '{endpoint_name}' is in status '{status}'. Waiting for it to become InService.")
            waiter = sagemaker_client.get_waiter('endpoint_in_service')
            waiter.wait(EndpointName=endpoint_name)
            logger.info(f"Endpoint '{endpoint_name}' is now InService.")
            st.session_state['endpoint_deployed'] = True
            return
        elif status == 'Failed':
            logger.info(f"Endpoint '{endpoint_name}' status is 'Failed'. Deleting and recreating.")
            sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
            waiter = sagemaker_client.get_waiter('endpoint_deleted')
            waiter.wait(EndpointName=endpoint_name)
            logger.info(f"Deleted failed endpoint '{endpoint_name}' successfully.")
            # Proceed to recreate
            describe_sagemaker_endpoint(api_keys)
            if create_endpoint(api_keys):
                st.session_state['endpoint_deployed'] = True
            else:
                st.error("Failed to create SageMaker endpoint.")
    except sagemaker_client.exceptions.ResourceNotFound:
        logger.info(f"Endpoint '{endpoint_name}' does not exist. Creating a new one.")
        # Create EndpointConfig if needed
        describe_sagemaker_endpoint(api_keys)
        # Now create the endpoint
        if create_endpoint(api_keys):
            st.session_state['endpoint_deployed'] = True
        else:
            st.error("Failed to create SageMaker endpoint.")
    except sagemaker_client.exceptions.ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'ValidationException':
            logger.info(f"Endpoint '{endpoint_name}' does not exist. Creating it now.")
            # Create EndpointConfig if needed
            describe_sagemaker_endpoint(api_keys)
            # Now create the endpoint
            if create_endpoint(api_keys):
                st.session_state['endpoint_deployed'] = True
            else:
                st.error("Failed to create SageMaker endpoint.")
        else:
            logger.error(f"Error checking SageMaker endpoint: {e}")
            st.error("An error occurred while checking the SageMaker endpoint status.")
            raise

def deploy_endpoint_async(api_keys):
    """
    Deploy the SageMaker endpoint in a separate thread to avoid blocking the main app.
    """
    if 'endpoint_deployment_started' not in st.session_state:
        st.session_state['endpoint_deployment_started'] = False

    if not st.session_state['endpoint_deployment_started']:
        st.session_state['endpoint_deployment_started'] = True
        deployment_thread = threading.Thread(target=check_and_deploy_sagemaker_endpoint, args=(api_keys,))
        deployment_thread.start()

def endpoint_idle_monitor(api_keys):
    global LAST_ENDPOINT_USAGE
    while True:
        time.sleep(60)  # Check every 60 seconds
        if LAST_ENDPOINT_USAGE:
            idle_time = time.time() - LAST_ENDPOINT_USAGE
            if idle_time > ENDPOINT_IDLE_TIMEOUT:
                logger.info(f"Endpoint idle for {idle_time} seconds. Deleting endpoint.")
                delete_sagemaker_endpoint(api_keys)
                LAST_ENDPOINT_USAGE = None
                break

def delete_sagemaker_endpoint(api_keys):
    try:
        sagemaker_client = boto3.client(
            'sagemaker',
            region_name=api_keys["aws_region"],
            aws_access_key_id=api_keys["aws_access_key"],
            aws_secret_access_key=api_keys["aws_secret_access_key"]
        )
        endpoint_name = api_keys["sentiment_endpoint_name"]

        # Check if the endpoint exists
        try:
            sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            # If no exception is raised, the endpoint exists
            logger.info(f"Endpoint '{endpoint_name}' exists. Proceeding to delete it.")
            sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
            logger.info(f"Successfully deleted endpoint '{endpoint_name}'.")
        except sagemaker_client.exceptions.ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'ValidationException' and 'Could not find endpoint' in e.response['Error']['Message']:
                # Endpoint does not exist
                logger.info(f"Endpoint '{endpoint_name}' does not exist. No deletion needed.")
            else:
                # Some other error occurred
                logger.error(f"Failed to delete endpoint '{endpoint_name}': {e}")
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

def default_converter(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f'Object of type {o.__class__.__name__} is not JSON serializable')

def save_sessions_to_s3(sessions, api_keys, username):
    s3_client = boto3.client(
        's3',
        aws_access_key_id=api_keys["aws_access_key"],
        aws_secret_access_key=api_keys["aws_secret_access_key"],
        region_name=api_keys["aws_region"]
    )
    bucket_name = api_keys["aws_bucket"]

    for session_id, session_data in sessions.items():
        # Prepare session data without vector_store
        session_data_copy = session_data.copy()
        vector_store = session_data_copy.pop('vector_store', None)

        # Serialize session data to JSON
        session_json = json.dumps(session_data_copy, default=default_converter)

        # Save session JSON to S3 under user-specific prefix
        session_key = f"sessions/{username}/{session_id}/session_data.json"
        s3_client.put_object(Bucket=bucket_name, Key=session_key, Body=session_json)

        # Save vector_store to S3
        if vector_store is not None:
            with tempfile.TemporaryDirectory() as temp_dir:
                vector_store.save_local(temp_dir)
                for filename in os.listdir(temp_dir):
                    file_path = os.path.join(temp_dir, filename)
                    s3_key = f"sessions/{username}/{session_id}/{filename}"
                    s3_client.upload_file(file_path, bucket_name, s3_key)
                    print(f"Uploaded {filename} to {s3_key}")

def load_sessions_from_s3(api_keys, username):
    s3_client = boto3.client(
        's3',
        aws_access_key_id=api_keys["aws_access_key"],
        aws_secret_access_key=api_keys["aws_secret_access_key"],
        region_name=api_keys["aws_region"]
    )
    bucket_name = api_keys["aws_bucket"]
    
    sessions = {}

    # List all sessions for the user in the bucket
    prefix = f"sessions/{username}/"
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    
    prefixes = set()
    for page in pages:
        for obj in page.get('Contents', []):
            key = obj['Key']
            parts = key.split('/')
            if len(parts) >= 4:
                session_id = parts[2]
                prefixes.add(session_id)

    for session_id in prefixes:
        # Load session data
        session_key = f"sessions/{username}/{session_id}/session_data.json"
        try:
            session_obj = s3_client.get_object(Bucket=bucket_name, Key=session_key)
            session_json = session_obj['Body'].read().decode('utf-8')
            session_data = json.loads(session_json)
        except Exception as e:
            logger.error(f"Failed to load session data for session {session_id}: {e}")
            continue

        # Load vector_store
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Download all vector store files for the session
                vector_store_prefix = f"sessions/{username}/{session_id}/"
                vector_store_objects = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=vector_store_prefix)
                
                if 'Contents' in vector_store_objects:
                    for obj in vector_store_objects['Contents']:
                        key = obj['Key']
                        filename = key.split('/')[-1]
                        file_path = os.path.join(temp_dir, filename)
                        s3_client.download_file(bucket_name, key, file_path)
                    
                    # Load FAISS vector store safely
                    embeddings = OpenAIEmbeddings(openai_api_key=api_keys["openai_api_key"])
                    session_data['vector_store'] = FAISS.load_local(
                        temp_dir, 
                        embeddings, 
                        allow_dangerous_deserialization=True  # Only enable this for trusted sources
                    )
                else:
                    session_data['vector_store'] = None
        except Exception as e:
            logger.warning(f"Skipping vector store for session {session_id}: {e}")
            session_data['vector_store'] = None

        sessions[session_id] = session_data

    return sessions

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
    embedding = response.data[0].embedding
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
    global LAST_ENDPOINT_USAGE
    
    try:
        LAST_ENDPOINT_USAGE = time.time()
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
    except sagemaker_runtime.exceptions.ValidationError as e:
        logger.error(f"Sentiment Analysis Failed: {e}")
        return 0.0
    except Exception as e:
        logger.error(f"Sentiment Analysis Failed: {e}")
        return 0.0

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
    # Assuming you have CSS content in a variable named 'css'
    st.markdown(css, unsafe_allow_html=True)

def get_location():
    location = st.query_params.get("location")
    if location:
        try:
            latitude, longitude = map(float, location[0].split(","))
            return {"latitude": latitude, "longitude": longitude}
        except ValueError:
            return None
    return None

# -----------------------------------------------------------------------------------
# 5. Authentication Functions
def login():
    st.header("Login")

    # Login Form
    with st.form(key='login_form', clear_on_submit=True):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_button = st.form_submit_button(label='Login')

    if login_button:
        if not username or not password:
            st.error("Please enter both username and password.")
        else:
            try:
                dynamodb = get_dynamodb_resource()
                users_table = dynamodb.Table('Users')

                # Retrieve user from DynamoDB
                response = users_table.get_item(Key={'username': username})
                user = response.get('Item')

                if user:
                    stored_password_hash = user['password_hash'].encode('utf-8')
                    if bcrypt.checkpw(password.encode('utf-8'), stored_password_hash):
                        st.session_state['authenticated'] = True
                        st.session_state['username'] = username
                        st.session_state['name'] = user['name']
                        st.success(f"Welcome, {user['name']}!")
                        st.rerun()
                    else:
                        st.error("Incorrect username or password.")
                else:
                    st.error("Incorrect username or password.")
            except Exception as e:
                logger.error(f"Error accessing DynamoDB: {e}")
                st.error("An error occurred during login. Please try again later.")

def register():
    st.header("Register")

    # Registration Form
    with st.form(key='register_form', clear_on_submit=True):
        username = st.text_input("Username")
        name = st.text_input("Full Name")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        register_button = st.form_submit_button(label='Register')

    if register_button:
        if not username or not name or not password or not confirm_password:
            st.error("Please fill out all fields.")
        elif password != confirm_password:
            st.error("Passwords do not match.")
        else:
            try:
                dynamodb = get_dynamodb_resource()
                users_table = dynamodb.Table('Users')

                # Check if username already exists
                response = users_table.get_item(Key={'username': username})
                if 'Item' in response:
                    st.error("Username already exists. Please choose a different one.")
                else:
                    # Hash the password
                    password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

                    # Add the new user to DynamoDB
                    users_table.put_item(
                        Item={
                            'username': username,
                            'name': name,
                            'password_hash': password_hash.decode('utf-8')
                        }
                    )
                    st.success("Registration successful! You can now log in.")
                    st.info("Please switch to the **Login** tab to access the chatbot.")
            except Exception as e:
                logger.error(f"Error registering user: {e}")
                st.error("An error occurred during registration. Please try again later.")

# -----------------------------------------------------------------------------------
# 6. Main Application
def main():
    username = st.session_state.get('username')
    name = st.session_state.get('name')
    
    # Load API keys
    api_keys = load_secrets()
    
    # Initialize OpenAI
    if api_keys["openai_api_key"] is None:
        st.error("OpenAI API key is not set. Please set it in the .env file.")
        return

    initialize_openai(api_keys["openai_api_key"])

    # Check and deploy SageMaker endpoint if needed
    try:
        deploy_endpoint_async(api_keys)
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

    # Start endpoint idle monitor thread
    threading.Thread(target=endpoint_idle_monitor, args=(api_keys,), daemon=True).start()

    # Initialize session state variables
    if 'sessions' not in st.session_state:
        st.session_state.sessions = load_sessions_from_s3(api_keys, st.session_state.get('username', ''))
    if 'selected_session' not in st.session_state:
        st.session_state.selected_session = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'doc_content' not in st.session_state:
        st.session_state.doc_content = ""
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'memory' not in st.session_state:
        st.session_state.memory = []

    # If no sessions exist, create a new one
    if not st.session_state.sessions:
        session_id = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        st.session_state.sessions[session_id] = {
            'name': f"Chat {session_id[-6:]}",
            'chat_history': [],
            'doc_content': "",
            'vector_store': None,
            'memory': []
        }
        st.session_state.selected_session = session_id

    # Sidebar: Chat Sessions (Dynamic Sidebar Integration)
    st.sidebar.header("Chat Sessions")

    selected_session = st.session_state.selected_session
    for session_id, session_data in st.session_state.sessions.items():
        session_name = session_data.get('name', f"Chat {session_id[-6:]}")
        label = session_name
        is_selected = session_id == selected_session

        with st.sidebar.expander(label, expanded=is_selected):
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Open", key=f"open_{session_id}"):
                    st.session_state.selected_session = session_id
                    st.session_state.chat_history = session_data.get('chat_history', [])
                    st.session_state.doc_content = session_data.get('doc_content', "")
                    st.session_state.vector_store = session_data.get('vector_store', None)
                    st.rerun()
            with col2:
                rename_clicked = st.button("Rename", key=f"rename_{session_id}")
                if rename_clicked:
                    new_name = st.text_input("New name", key=f"new_name_{session_id}")
                    if new_name:
                        st.session_state.sessions[session_id]['name'] = new_name
                        # Save the updated sessions to S3
                        save_sessions_to_s3(st.session_state.sessions, api_keys, username)
                        st.rerun()
            with col3:
                if st.button("Delete", key=f"delete_{session_id}"):
                    del st.session_state.sessions[session_id]
                    st.session_state.selected_session = None
                    st.session_state.chat_history = []
                    # Save the updated sessions to S3
                    save_sessions_to_s3(st.session_state.sessions, api_keys, username)
                    st.rerun()
    
    # New Chat Button
    if st.sidebar.button("New Chat"):
        # Create a new session
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
        save_sessions_to_s3(st.session_state.sessions, api_keys, username)
        st.rerun()
    
    if st.sidebar.button("Logout"):
        save_sessions_to_s3(st.session_state.sessions, api_keys, username)
        st.session_state.clear()  # Clear all session state
        st.rerun()
    
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
        st.subheader(f"Welcome, {st.session_state.get('name', 'User')}!")

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

        # Create a placeholder for the typing indicator
        typing_placeholder = st.empty()
        typing_placeholder.markdown(
            "<div class='bubble assistant'>Bot is typing...</div>", 
            unsafe_allow_html=True
        )

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

        # Determine response based on input context
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

        # Remove the typing indicator
        typing_placeholder.empty()

        # Save the bot response in the chat history
        st.session_state.chat_history.append(("assistant", response))

        # Save the question, response, and embedding in the session memory
        st.session_state.memory.append({
            'question': user_input,
            'response': response,
            'embedding': new_question_embedding,
            'timestamp': current_time.isoformat()
        })

        # Update session data
        session_data = st.session_state.sessions.get(st.session_state.selected_session, {})
        session_data['chat_history'] = st.session_state.chat_history
        session_data['doc_content'] = st.session_state.doc_content
        session_data['vector_store'] = st.session_state.vector_store
        session_data['memory'] = st.session_state.memory

        # Generate a new session name if needed
        if session_data['name'].startswith("Chat "):
            session_name = generate_session_name(st.session_state.chat_history)
            if session_name:
                session_data['name'] = session_name
                st.rerun()

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

    # Save sessions to S3 on exit
    def save_sessions_callback():
        logger.info("Application is exiting. Saving sessions to S3.")
        username = st.session_state.get('username', '')
        if 'sessions' in st.session_state and username:
            try:
                save_sessions_to_s3(st.session_state.sessions, api_keys, username)
                logger.info("Sessions saved successfully.")
            except Exception as e:
                logger.error(f"Failed to save sessions on exit: {e}")
        else:
            logger.info("No sessions to save or user not logged in.")

    # Delete SageMaker endpoint on exit
    def on_exit():
        logger.info("Application is exiting. Deleting SageMaker endpoint.")
        delete_sagemaker_endpoint(api_keys)
        save_sessions_callback()
    atexit.register(on_exit)

# -----------------------------------------------------------------------------------
# 7. Run the Application
if __name__ == "__main__":
    # Authentication Check
    if 'authenticated' not in st.session_state or not st.session_state['authenticated']:
        # Display Login and Register forms in the main container using Tabs
        tabs = st.tabs(["Login", "Register"])

        with tabs[0]:
            login()

        with tabs[1]:
            register()
    else:
        main()
