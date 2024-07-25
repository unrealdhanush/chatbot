# Personal Assistant Chatbot

This is a personal project to create a chatbot using Streamlit, OpenAI's GPT-3.5, and other APIs. The project is designed for learning and understanding purposes and is available freely for the public under the MIT License.

## Features

- Chat with a personal assistant
- Upload and process documents (PDF, DOCX, TXT)
- Search the web using SerpAPI
- Get weather information using OpenWeatherMap API

## Setup

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/your-repo-name.git
    cd your-repo-name
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the dependencies:
    ```sh
    pip install -r requirements.txt
    ```

4. Create a `.env` file based on `.env.example` and fill in your API keys.

5. Run the Streamlit app:
    ```sh
    streamlit run src/chatbot.py
    ```

## Usage
Open your browser and go to `http://localhost:8501` to interact with the chatbot.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
