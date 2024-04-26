import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

def get_completion(prompt):
    _ = load_dotenv(find_dotenv())
    messages = [{"role": "user", "content": prompt}]
    client = OpenAI(
    api_key=os.environ.get("OPEN_API_KEY"),
    )
    response = client.chat.completions.create(
    messages=messages,
    model="gpt-3.5-turbo",
    )
    return response.choices[0].message.content


customer_email = """Arrr, I be fuming that me blender lid flew off and splattered me kitchen walls with smoothie! And to make matters worse, the warranty don't cover the cost of cleaning up me kitchen. I need yer help right now, matey!"""

style = """American English in a calm, kind and respectful tone"""

prompt = f"""Translate the text that is delimited by triple backticks into a style that is {style}. text: ```{customer_email}```"""

print(get_completion(prompt))