import os
import requests
import sqlite3
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from serpapi import GoogleSearch
import datetime
from PyPDF2 import PdfReader
from docx import Document
import io
import re

# Load API keys from .env file
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ.get("OPEN_API_KEY")
serpapi_api_key = os.environ.get("SERP_API_KEY")
weather_api_key = os.environ.get("WEATHER_API_KEY")

# Initialize OpenAI
client = OpenAI(api_key=openai_api_key)

# Get the base directory and set the data directory path
base_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(base_dir)
data_dir = os.path.join(project_dir, 'data')
db_path = os.path.join(data_dir, 'chatbot_memory.db')

conn = sqlite3.connect(db_path)
c = conn.cursor()

# Create tables for storing data
c.execute('''CREATE TABLE IF NOT EXISTS memory (question TEXT, response TEXT)''')
c.execute('''CREATE TABLE IF NOT EXISTS chat_sessions (session_id TEXT PRIMARY KEY, name TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
c.execute('''CREATE TABLE IF NOT EXISTS chat_messages (session_id TEXT, role TEXT, content TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
c.execute('''CREATE TABLE IF NOT EXISTS documents (session_id TEXT, filename TEXT, content TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
conn.commit()

# Function to process uploaded files
def process_uploaded_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(uploaded_file)
        return "\n".join([para.text for para in doc.paragraphs])
    else:
        return uploaded_file.getvalue().decode("utf-8")

# Function to truncate text to fit within token limits
def truncate_text(text, max_tokens=4000):
    words = text.split()
    if len(words) > max_tokens:
        return ' '.join(words[:max_tokens])
    return text

# Function to create a prompt for OpenAI
def create_prompt(user_input, doc_content):
    truncated_doc_content = truncate_text(doc_content)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_input},
        {"role": "system", "content": f"Here is some additional context from the uploaded documents:\n{truncated_doc_content}"}
    ]
    return messages

# Define the function to get completion from OpenAI API
def get_completion(prompt, model="gpt-3.5-turbo"):
    response = client.chat.completions.create(
        model=model,
        messages=prompt,
    )
    return response.choices[0].message.content

# Define the function to search the web using SerpAPI
def search_web(query):
    search = GoogleSearch({"q": query, "api_key": serpapi_api_key})
    results = search.get_dict()
    return results.get("organic_results", [])

# Define the function to get weather information
def get_weather(location="London"):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={weather_api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        weather_description = data['weather'][0]['description']
        temperature = data['main']['temp']
        return f"The weather in {location} is {weather_description} with a temperature of {temperature}°C."
    else:
        return "Sorry, I couldn't retrieve the weather information."

# Function to load chat history
def load_chat_history(session_id):
    c.execute("SELECT role, content FROM chat_messages WHERE session_id = ? ORDER BY timestamp ASC", (session_id,))
    return c.fetchall()

# Function to load document content
def load_document_content(session_id):
    c.execute("SELECT content FROM documents WHERE session_id = ?", (session_id,))
    rows = c.fetchall()
    return "\n".join([row[0] for row in rows])

# Streamlit UI
st.set_page_config(layout="wide")

# Sidebar for chat sessions
st.sidebar.title("Chat Sessions")

# Function to create a new chat session
def create_new_session():
    session_id = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    c.execute("INSERT INTO chat_sessions (session_id, name) VALUES (?, ?)", (session_id, f"Chat {session_id[-6:]}"))
    conn.commit()
    return session_id

# Load chat sessions
c.execute("SELECT session_id, name, timestamp FROM chat_sessions ORDER BY timestamp DESC")
sessions = c.fetchall()

# Display chat sessions in the sidebar with better styling
session_ids = [session[0] for session in sessions]
session_names = [session[1] for session in sessions]
session_timestamps = [session[2] for session in sessions]

# Initialize selected_session in session state if not already present
if 'selected_session' not in st.session_state:
    st.session_state.selected_session = session_ids[0] if session_ids else None

# Function to display the sidebar menu with chat sessions
def display_sidebar_menu():
    selected_session = st.session_state.selected_session
    for i, session_id in enumerate(session_ids):
        label = session_names[i] if session_names[i] else f"Chat {session_id[-6:]}"
        is_selected = session_id == selected_session
        with st.sidebar.expander(label, expanded=is_selected):
            if st.button("Open", key=f"open_{session_id}"):
                st.session_state.selected_session = session_id
                st.session_state.chat_history = load_chat_history(session_id)
                st.session_state.doc_content = load_document_content(session_id)
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

# Display the sidebar menu
display_sidebar_menu()

# Button to create a new session
if st.sidebar.button("New Chat"):
    st.session_state.selected_session = create_new_session()
    st.session_state.chat_history = []
    st.rerun()

# Load chat messages for the selected session
if st.session_state.selected_session:
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = load_chat_history(st.session_state.selected_session)
    if 'doc_content' not in st.session_state:
        st.session_state.doc_content = load_document_content(st.session_state.selected_session)

# Add CSS for the text bubbles and dynamic theme support
st.markdown(
    """
    <style>
    .main {
        color: var(--text-color);
    }
    .stTextInput>div>div>input {
        color: var(--input-text-color);
        background-color: var(--input-bg-color);
    }
    .stButton>button {
        background-color: var(--button-bg-color);
        color: var(--button-text-color);
    }
    .bubble-container {
        display: flex;
        align-items: flex-end;
        margin-bottom: 10px;
    }
    .bubble {
        max-width: 70%;
        padding: 10px;
        border-radius: 20px;
        margin: 5px;
        position: relative;
    }
    .bubble.user {
        background-color: #007bff;
        color: white;
        margin-left: auto;
        order: 1;
    }
    .bubble.assistant {
        background-color: #f1f1f1;
        color: black;
        order: 2;
    }
    .icon {
        width: 30px;
        height: 30px;
        border-radius: 50%;
        margin: 5px;
    }
    .icon.user {
        order: 2;
    }
    .icon.assistant {
        order: 1;
    }
    </style>
    """, unsafe_allow_html=True)

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
    # Add the user's message to chat history immediately
    st.session_state.chat_history.append(("user", user_input))
    c.execute("INSERT INTO chat_messages (session_id, role, content) VALUES (?, ?, ?)", (st.session_state.selected_session, "user", user_input))
    conn.commit()

    # Process uploaded file
    if uploaded_file:
        file_content = process_uploaded_file(uploaded_file)
        st.session_state.doc_content = file_content
        c.execute("INSERT INTO documents (session_id, filename, content) VALUES (?, ?, ?)",
                  (st.session_state.selected_session, uploaded_file.name, file_content))
        conn.commit()

    # Check if the question is in the memory database
    c.execute("SELECT response FROM memory WHERE question = ?", (user_input,))
    row = c.fetchone()
    if row:
        response = row[0]
    else:
        if "weather" in user_input.lower():
            location = user_input.split("in")[-1].strip() if "in" in user_input.lower() else "London"
            response = get_weather(location)
        elif "search" in user_input.lower():
            search_results = search_web(user_input)
            response = "\n".join([f"**{result['title']}**\n{result['link']}\n{result['snippet']}\n" for result in search_results])
        else:
            prompt = create_prompt(user_input, st.session_state.doc_content)
            response = get_completion(prompt, model="gpt-3.5-turbo")

        # Save the question and response in the memory database
        c.execute("INSERT INTO memory (question, response) VALUES (?, ?)", (user_input, response))
        conn.commit()

    # Update chat history with the assistant's response
    st.session_state.chat_history.append(("assistant", response))
    c.execute("INSERT INTO chat_messages (session_id, role, content) VALUES (?, ?, ?)", (st.session_state.selected_session, "assistant", response))
    conn.commit()

# Display chat history when a session is selected
if st.session_state.selected_session and st.session_state.chat_history:
    with conversation_container:
        for role, content in st.session_state.chat_history:
            role_class = 'user' if role == 'user' else 'assistant'
            icon_url = "https://img.icons8.com/ios-filled/50/000000/user-male-circle.png" if role == 'user' else "https://img.icons8.com/fluency-systems-filled/48/bot.png"
            st.markdown(
                f'''
                <div class="bubble-container">
                    <div class="bubble {role_class}">{content}</div>
                    <img src="{icon_url}" class="icon {role_class}" />
                </div>
                ''',
                unsafe_allow_html=True
            )

conn.close()
