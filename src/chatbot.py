import os
import requests
import sqlite3
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from serpapi import GoogleSearch

# Load API keys from .env file
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ.get("OPEN_API_KEY")
serpapi_api_key = os.environ.get("SERP_API_KEY")
weather_api_key = os.environ.get("WEATHER_API_KEY")

# Initialize OpenAI
client = OpenAI(api_key=openai_api_key)

# Initialize database connection
base_dir = os.path.dirname(__file__)
project_dir = os.path.dirname(base_dir)
data_dir = os.path.join(project_dir, 'data')
db_path = os.path.join(data_dir, 'chatbot_memory.db')

# Create the data directory if it doesn't exist
os.makedirs(data_dir, exist_ok=True)

conn = sqlite3.connect(db_path)
c = conn.cursor()

# Create a table to store user preferences or frequently asked questions
c.execute('''CREATE TABLE IF NOT EXISTS memory
             (question TEXT, response TEXT)''')
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

# Streamlit UI
st.set_page_config(layout="wide")

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
        flex-direction: column;
        align-items: flex-start;
        margin-bottom: 10px;
    }
    .bubble {
        max-width: 70%;
        padding: 10px;
        border-radius: 20px;
        margin: 5px 0;
        position: relative;
    }
    .bubble.user {
        align-self: flex-end;
        background-color: #007bff;
        color: white;
    }
    .bubble.assistant {
        align-self: flex-start;
        background-color: #f1f1f1;
        color: black;
    }
    .icon {
        width: 30px;
        height: 30px;
        border-radius: 50%;
        position: absolute;
        top: -20px;
    }
    .icon.user {
        left: -40px;
    }
    .icon.assistant {
        right: -40px;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state to store chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

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
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Display chat history
    with conversation_container:
        for chat in st.session_state.chat_history:
            role_class = 'user' if chat['role'] == 'user' else 'assistant'
            icon_url = "https://img.icons8.com/ios-filled/50/000000/user-male-circle.png" if chat['role'] == 'user' else "https://img.icons8.com/fluency-systems-filled/48/bot.png"
            st.markdown(
                f'''
                <div class="bubble-container">
                    <img src="{icon_url}" class="icon {role_class}" />
                    <div class="bubble {role_class}">{chat["content"]}</div>
                </div>
                ''',
                unsafe_allow_html=True
            )

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
    st.session_state.chat_history.append({"role": "assistant", "content": response})

    # Display only the assistant's response
    with conversation_container:
        role_class = 'assistant'
        icon_url = "https://img.icons8.com/fluency-systems-filled/48/bot.png"
        st.markdown(
            f'''
            <div class="bubble-container">
                <img src="{icon_url}" class="icon {role_class}" />
                <div class="bubble {role_class}">{response}</div>
            </div>
            ''',
            unsafe_allow_html=True
        )

# Close the database connection
conn.close()