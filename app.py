
import gradio as gr
from openai import OpenAI
import os
import getpass
import pandas as pd
import numpy as np
import math

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.output_parsers import StrOutputParser

# --- Set up OpenAI API Key --- #
# For deployment, it's best to use environment variables for API keys
# You would set OPENAI_API_KEY as a secret in Hugging Face Spaces
# For local testing, you might still use getpass or load from a .env file

# Try to get API key from environment, if not present, prompt (for local testing)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
if OPENAI_API_KEY is None:
    # This part is primarily for local Colab testing if you run app.py directly
    # On Hugging Face Spaces, it will be loaded from secrets
    OPENAI_API_KEY = getpass.getpass("Enter your OpenAI API key: ")

client = OpenAI(api_key=OPENAI_API_KEY)

# --- Data Preparation (from your notebook) ---
df = pd.read_csv("/content/drive/MyDrive/datasets/ProductChatbot/train.csv", index_col=0, engine='python', on_bad_lines='warn')

product_description = []
for row in df.iterrows():
    product = ""
    title = row[1]["TITLE"]
    if type(title) != float or not math.isnan(title):
        product += "Title\n" + title + "\n"
    description = row[1]["DESCRIPTION"]
    if type(description) != float or not math.isnan(description):
        product += "Description\n" + description + "\n"

    # Only append if either title or description has content
    if title or description:
        product = product.strip()
        product_description.append(product)

# --- Vector Store Setup (from your notebook) ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)
documents = text_splitter.create_documents(product_description)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vector = FAISS.from_documents(documents, embeddings)

# --- Chatbot Building (from your notebook) ---
llm = ChatOpenAI(api_key=OPENAI_API_KEY, model='gpt-4o-mini')
output_parser = StrOutputParser()
prompt_template = ChatPromptTemplate.from_template(
    """Answer the following question based only on the provided context:\n\n    <context>\n    {context}\n    </context>\n\n    Question: {input}""",
    output_parser=output_parser
)
document_chain = create_stuff_documents_chain(llm, prompt_template)

retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# --- Final Response Function ---
def final_response(user_query):
    response = retrieval_chain.invoke({"input": user_query})['answer']
    
    # This second call to OpenAI to 'format' the response is less ideal for a deployed app
    # as it incurs double cost. A better approach would be to refine the initial prompt
    # or post-process the answer directly in Python if the formatting is simple.
    # For this example, we'll keep it as in the notebook.
    prompt_format = f"Format the responses properly in {response}. Just return the product names, no other text"

    openai_response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{'role': 'user', 'content': prompt_format}]
    )
    return openai_response.choices[0].message.content

# --- Gradio Interface ---
app = gr.Interface(
    fn=final_response,
    inputs=gr.Textbox(lines=2, label="Enter your query"),
    outputs=gr.Textbox(lines=10, label="Recommendation"),
    title="Review Genie",
    description="Type your question below to get the recommendations",
    theme="Ocean",
    allow_flagging="never"
)

if __name__ == "__main__":
    app.launch()
