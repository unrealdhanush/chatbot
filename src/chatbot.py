import os
import requests
import sqlite3
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from serpapi import GoogleSearch
import datetime

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

# Create a table to store user preferences or frequently asked questions
c.execute('''CREATE TABLE IF NOT EXISTS memory
             (question TEXT, response TEXT)''')
conn.commit()

# Create a table to store chat sessions
c.execute('''CREATE TABLE IF NOT EXISTS chat_sessions
             (session_id TEXT PRIMARY KEY, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
conn.commit()

# Create a table to store chat messages
c.execute('''CREATE TABLE IF NOT EXISTS chat_messages
             (session_id TEXT, role TEXT, content TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
conn.commit()

# Prompt Creator
def create_prompt(user_input):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_input}
    ]
    return messages

# Define the function to get completion from OpenAI API
def get_completion(prompt, model="gpt-3.5-turbo"):
    response = client.chat.completions.create(
        model=model,
        messages=create_prompt(prompt),
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

# Streamlit UI
st.set_page_config(layout="wide")

# Sidebar for chat sessions
st.sidebar.title("Chat Sessions")

# Function to create a new chat session
def create_new_session():
    session_id = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    c.execute("INSERT INTO chat_sessions (session_id) VALUES (?)", (session_id,))
    conn.commit()
    return session_id

# Load chat sessions
c.execute("SELECT session_id, timestamp FROM chat_sessions ORDER BY timestamp DESC")
sessions = c.fetchall()

# Display chat sessions in the sidebar with better styling
session_ids = [session[0] for session in sessions]
session_timestamps = [session[1] for session in sessions]
session_labels = [f"Chat {i+1}: {session_timestamps[i]}" for i in range(len(sessions))]

# Initialize selected_session in session state if not already present
if 'selected_session' not in st.session_state:
    st.session_state.selected_session = session_ids[0] if session_ids else None

# Add CSS for better sidebar styling
st.sidebar.markdown(
    """
    <style>
    .sidebar .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .sidebar .block-container .stButton {
        margin-bottom: 1rem;
    }
    .session-button {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px;
        border: 1px solid #e6e6e6;
        border-radius: 5px;
        margin-bottom: 10px;
        cursor: pointer;
    }
    .session-button:hover {
        background-color: red;
    }
    .session-button.selected {
        background-color: gray;
        border: 2px solid red;
        color: white;
    }
    .session-button span {
        flex-grow: 1;
    }
    .session-button button {
        background: none;
        border: none;
        color: #007bff;
        cursor: pointer;
        font-size: 14px;
    }
    .session-button.selected button {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def display_session_button(session_id, label, is_selected):
    button_class = "session-button selected" if is_selected else "session-button"
    clicked = st.sidebar.button(label, key=session_id)
    if clicked:
        st.session_state.selected_session = session_id
        st.session_state.chat_history = load_chat_history(session_id)
        st.rerun()

# Display session buttons
for i, session_id in enumerate(session_ids):
    label = session_labels[i]
    is_selected = session_id == st.session_state.selected_session
    display_session_button(session_id, label, is_selected)

# Button to create a new session
if st.sidebar.button("New Chat"):
    st.session_state.selected_session = create_new_session()
    st.session_state.chat_history = []
    st.rerun()

# Load chat messages for the selected session
if st.session_state.selected_session:
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = load_chat_history(st.session_state.selected_session)

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
        submit_button = st.form_submit_button(label='Ask')

if submit_button and user_input:
    # Add the user's message to chat history immediately
    st.session_state.chat_history.append(("user", user_input))
    c.execute("INSERT INTO chat_messages (session_id, role, content) VALUES (?, ?, ?)", (st.session_state.selected_session, "user", user_input))
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
            response = get_completion(user_input, model="gpt-3.5-turbo")

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
