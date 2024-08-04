css = """
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
    """
    