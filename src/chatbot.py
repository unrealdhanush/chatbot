import os
import requests
import sqlite3
import streamlit as st
import openai
import re
import speech_recognition as sr
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from dotenv import load_dotenv, find_dotenv
from serpapi import GoogleSearch
import datetime
from PyPDF2 import PdfReader
from docx import Document
import boto3
from css import css
from js import js
import tiktoken
import pyttsx3
from textblob import TextBlob
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity 

CACHE_EXPIRY_TIME = 60 * 10  # Cache expiry time in seconds (e.g., 10 minutes for weather data)

def load_api_keys():
    _ = load_dotenv(find_dotenv())
    return {
        "openai_api_key": os.environ.get("OPENAI_API_KEY"),
        "serpapi_api_key": os.environ.get("SERP_API_KEY"),
        "weather_api_key": os.environ.get("WEATHER_API_KEY"),
        "aws_access_key": os.environ.get("AWS_ACCESS_KEY"),
        "aws_secret_access_key": os.environ.get("AWS_SECRET_ACCESS_KEY"),
        "aws_bucket": os.environ.get("AWS_BUCKET")
    }

def initialize_openai(api_key):
    openai.api_key = api_key

def initialize_db(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    # Create tables if not exist
    c.execute('''CREATE TABLE IF NOT EXISTS memory (question TEXT, response TEXT, embedding BLOB, timestamp DATETIME)''')
    c.execute('''CREATE TABLE IF NOT EXISTS chat_sessions (session_id TEXT PRIMARY KEY, name TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS chat_messages (session_id TEXT, role TEXT, content TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS documents (session_id TEXT, filename TEXT, content TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    # Check if 'embedding' and 'timestamp' columns exist in memory table, if not, add them
    c.execute("PRAGMA table_info(memory)")
    columns = [col[1] for col in c.fetchall()]
    if 'embedding' not in columns:
        c.execute('ALTER TABLE memory ADD COLUMN embedding BLOB')
    if 'timestamp' not in columns:
        c.execute('ALTER TABLE memory ADD COLUMN timestamp DATETIME')
    conn.commit()
    return conn, c

def save_file_to_s3(file, aws_keys):
    s3 = boto3.client('s3', aws_access_key_id=aws_keys["aws_access_key"], aws_secret_access_key=aws_keys["aws_secret_access_key"])
    file.seek(0)
    s3.upload_fileobj(file, aws_keys["aws_bucket"], file.name)

def get_embedding(text):
    response = openai.embeddings.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    embedding = response.data[0].embedding
    return embedding

def find_similar_questions(embedding, c, top_k=3):
    c.execute("SELECT question, response, embedding FROM memory")
    rows = c.fetchall()
    similarities = []
    for question, response, stored_embedding_blob in rows:
        if stored_embedding_blob is None:
            continue  # Skip entries without embeddings
        stored_embedding = np.frombuffer(stored_embedding_blob, dtype=np.float32)
        sim = cosine_similarity([embedding], [stored_embedding])[0][0]
        similarities.append((sim, question, response))
    # Sort by similarity
    similarities.sort(key=lambda x: x[0], reverse=True)
    return similarities[:top_k]

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

def count_tokens(text, model="gpt-3.5-turbo"):
    tokenizer = tiktoken.encoding_for_model(model)
    tokens = tokenizer.encode(text)
    return len(tokens)

def truncate_text(text, max_tokens):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[-max_tokens:]
    return tokenizer.decode(tokens)

def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity  # Returns a value between -1.0 and 1.0

def get_response(user_input, doc_content, chat_history, similar_past_interactions):
    # Including last N messages for context
    context_messages = [{"role": role, "content": content} for role, content in chat_history[-5:]]
    user_message = {"role": "user", "content": user_input}

    user_sentiment = analyze_sentiment(user_input)
    if user_sentiment < -0.3:
        system_prompt = "You are a compassionate assistant. Provide a thoughtful and empathetic response to the user's concerns."
    else:
        system_prompt = "You are a helpful assistant. Provide a thoughtful response to the user's concerns."

    # Prepare past interactions context
    past_interactions_text = "\n".join(
        [f"User: {q}\nAssistant: {r}" for _, q, r in similar_past_interactions]
    )
    # Modify system_message to include past interactions
    system_message = {
        "role": "system",
        "content": f"{system_prompt}\n\nHere are some past interactions that might help:\n{past_interactions_text}"
    }

    if doc_content:
        text_chunks = get_text_chunks(doc_content)
        vector_store = get_vector_store(text_chunks)
        relevant_chunks = get_relevant_chunks(user_input, vector_store)
        context = "\n".join(relevant_chunks)
        # Adjust max_tokens to account for context messages
        total_allowed_tokens = 4096 - 150  # Model's max tokens minus response tokens
        context_tokens = count_tokens("\n".join([msg["content"] for msg in context_messages]))
        available_tokens = total_allowed_tokens - context_tokens
        truncated_context = truncate_text(context, available_tokens)
        context_message = {"role": "system", "content": f"Context:\n{truncated_context}"}
        messages = [system_message, context_message] + context_messages + [user_message]
    else:
        messages = [system_message] + context_messages + [user_message]

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=150
    )
    return response.choices[0].message.content.strip()

def search_web(query, serpapi_api_key):
    search = GoogleSearch({"q": query, "api_key": serpapi_api_key})
    results = search.get_dict()
    return results.get("organic_results", [])

def get_audio_input():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening...")
        audio = r.listen(source)
    try:
        text = r.recognize_google(audio)
        st.success(f"You said: {text}")
        return text
    except sr.UnknownValueError:
        st.error("Sorry, I could not understand the audio.")
    except sr.RequestError as e:
        st.error(f"Could not request results; {e}")
    return None

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

def load_chat_history(c, session_id):
    c.execute("SELECT role, content FROM chat_messages WHERE session_id = ? ORDER BY timestamp ASC", (session_id,))
    return c.fetchall()

def load_document_content(c, session_id):
    c.execute("SELECT content FROM documents WHERE session_id = ?", (session_id,))
    rows = c.fetchall()
    return "\n".join([row[0] for row in rows])

def create_new_session(c, conn):
    session_id = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    c.execute("INSERT INTO chat_sessions (session_id, name) VALUES (?, ?)", (session_id, f"Chat {session_id[-6:]}"))
    conn.commit()
    return session_id

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

def display_sidebar_menu(c, sessions, conn):
    selected_session = st.session_state.selected_session
    for i, session in enumerate(sessions):
        session_id, session_name, timestamp = session
        label = session_name if session_name else f"Chat {session_id[-6:]}"
        is_selected = session_id == selected_session
        with st.sidebar.expander(label, expanded=is_selected):
            if st.button("Open", key=f"open_{session_id}"):
                st.session_state.selected_session = session_id
                st.session_state.chat_history = load_chat_history(c, session_id)
                st.session_state.doc_content = load_document_content(c, session_id)
                st.rerun()
            if st.button("Rename", key=f"rename_{session_id}"):
                st.session_state.rename_session_id = session_id
                st.session_state.rename_input = ""
                st.rerun()
            if 'rename_session_id' in st.session_state and st.session_state.rename_session_id == session_id:
                st.session_state.rename_input = st.text_input("New name", key=f"new_name_{session_id}")
                if st.button("Save", key=f"save_{session_id}"):
                    new_name = st.session_state.rename_input
                    if new_name:
                        c.execute("UPDATE chat_sessions SET name = ? WHERE session_id = ?", (new_name, session_id))
                        conn.commit()
                        del st.session_state.rename_session_id
                        st.rerun()
            if st.button("Delete", key=f"delete_{session_id}"):
                c.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))
                conn.commit()
                st.session_state.selected_session = None
                st.session_state.chat_history = []
                st.rerun()

def add_css_and_html():
    st.markdown(css, unsafe_allow_html=True)

def get_location():
    query_params = st.query_params
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

    # Initialize database
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    data_dir = os.path.join(project_dir, 'data')
    db_path = os.path.join(data_dir, 'chatbot_memory.db')
    conn, c = initialize_db(db_path)

    # Load chat sessions
    c.execute("SELECT session_id, name, timestamp FROM chat_sessions ORDER BY timestamp DESC")
    sessions = c.fetchall()

    # Initialize selected_session in session state if not already present
    if 'selected_session' not in st.session_state:
        st.session_state.selected_session = sessions[0][0] if sessions else None

    # Display the sidebar menu with chat sessions
    display_sidebar_menu(c, sessions, conn)

    # Button to create a new session
    if st.sidebar.button("New Chat"):
        st.session_state.selected_session = create_new_session(c, conn)
        st.session_state.chat_history = []
        st.session_state.doc_content = ""
        st.rerun()

    # Load chat messages and document content for the selected session
    if st.session_state.selected_session:
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = load_chat_history(c, st.session_state.selected_session)
        if 'doc_content' not in st.session_state:
            st.session_state.doc_content = load_document_content(c, st.session_state.selected_session)

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
        c.execute("INSERT INTO chat_messages (session_id, role, content) VALUES (?, ?, ?)",
                  (st.session_state.selected_session, "user", user_input))
        conn.commit()

        # Process uploaded file
        if uploaded_file:
            file_content = process_uploaded_file(uploaded_file)
            st.session_state.doc_content += file_content
            c.execute("INSERT INTO documents (session_id, filename, content) VALUES (?, ?, ?)",
                      (st.session_state.selected_session, uploaded_file.name, file_content))
            conn.commit()
            # Save the file to S3
            save_file_to_s3(uploaded_file, api_keys)

        # Compute embedding for the new question
        new_question_embedding = get_embedding(user_input)

        # Retrieve similar past interactions
        similar_past_interactions = find_similar_questions(new_question_embedding, c)

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
                similar_past_interactions
            )

        # Save the question, response, and embedding in the memory database
        embedding_bytes = np.array(new_question_embedding, dtype=np.float32).tobytes()
        c.execute(
            "INSERT INTO memory (question, response, embedding, timestamp) VALUES (?, ?, ?, ?)",
            (user_input, response, embedding_bytes, current_time.isoformat())
        )
        conn.commit()

        # Update chat history with the assistant's response
        st.session_state.chat_history.append(("assistant", response))
        c.execute("INSERT INTO chat_messages (session_id, role, content) VALUES (?, ?, ?)",
                  (st.session_state.selected_session, "assistant", response))
        conn.commit()
        c.execute("SELECT name FROM chat_sessions WHERE session_id = ?", (st.session_state.selected_session,))
        current_session_name = c.fetchone()[0]
        if current_session_name.startswith("Chat "):
            # Generate a new session name
            session_name = generate_session_name(st.session_state.chat_history)
            if session_name:
                c.execute("UPDATE chat_sessions SET name = ? WHERE session_id = ?",
                          (session_name, st.session_state.selected_session))
                conn.commit()
                # Update the sessions list and rerun to refresh the sidebar
                c.execute("SELECT session_id, name, timestamp FROM chat_sessions ORDER BY timestamp DESC")
                sessions = c.fetchall()
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

    conn.close()

if __name__ == "__main__":
    main()
