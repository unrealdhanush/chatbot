# Personal Assistant Chatbot

This is a personal project to create a chatbot using Streamlit, OpenAI's GPT-3.5, and other APIs. The project is designed for learning and understanding purposes and is available freely for the public under the MIT License.

## Features

- Chat with a personal assistant
- Upload and process documents (PDF, DOCX, TXT)
- Search the web using SerpAPI
- Get weather information using OpenWeatherMap API
- Implement Retrieval-Augmented Generation (RAG) for enhanced responses
- Sentiment analysis using Amazon SageMaker and Hugging Face models.
- Session management using AWS S3 and DynamoDB for user data storage.
- Authentication (login and registration) using DynamoDB.

## Retrieval-Augmented Generation (RAG)

RAG is a technique that combines retrieval-based and generation-based methods to provide more accurate and contextually relevant responses. In this project, RAG is implemented to enhance the chatbot's ability to generate responses based on both the user's input and the additional context provided by uploaded documents.

## Setup
### Prerequisites
- Python 3.8+
- AWS account for SageMaker and S3 integration
- API keys for OpenAI, SerpAPI, OpenWeatherMap, and AWS credentials

### Steps to Set Up
1. Clone the repository:
    ```sh
    git clone https://github.com/unrealdhanush/chatbot.git
    cd chatbot
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

1. **Authentication:** Users must either register or log in via the login/register forms to access the chatbot interface. The user data is stored in DynamoDB.

2. **Chatting with the assistant:** Once logged in, users can chat with the assistant by typing their queries. The assistant uses OpenAI GPT-3.5 for generating responses.

3. **Document processing:** Users can upload PDF, DOCX, or TXT files. The content of the uploaded files will be split into chunks and embedded for retrieval-based question answering.

4. **Web search:** The chatbot can search the web using SerpAPI and return relevant results.

5. **Weather information:** By providing location details, users can ask for weather updates using the OpenWeatherMap API.

6. **Session management:** The chat history, uploaded documents, and embeddings are saved per session. Sessions are stored in AWS S3 and can be retrieved for continuity.

## API Integrations
- **OpenAI GPT-3.5:** For generating responses and embeddings.
- **SerpAPI:** For web searches. (Currently not active, but you can activate)
- **OpenWeatherMap:** For weather information. (Currently not active, but you can activate)
- **AWS (S3, DynamoDB, SageMaker):** For session management, sentiment analysis, and storing user data.

## Logs and Monitoring
- The project includes a logging system that records daily, weekly, and monthly logs to a `logs/` directory. This is helpful for monitoring chatbot interactions and tracking errors.
- Log files are handled via `TimedRotatingFileHandler` to create backups.

## AWS SageMaker Endpoint
The chatbot uses a SageMaker-hosted Hugging Face sentiment analysis model. The endpoint is deployed asynchronously to avoid blocking the main app, and it monitors for idle time to delete the endpoint if unused.

## Deploying SageMaker Sentiment Analysis Endpoint
The app can automatically deploy or update the endpoint if it is not already active. It uses a DistilBERT model fine-tuned for sentiment analysis, hosted on AWS SageMaker. (Recommended: Create endpoint manually and deactivate `create_endpoint` and `delete_endpoint` function if using in production environments)

## Launch

Open your browser and go to `http://localhost:8501` to interact with the chatbot.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
