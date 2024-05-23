import os
import streamlit as st
from typing import Generator
from groq import Groq
from dotenv import load_dotenv
import yaml
import subprocess
import re

# Load environment variables from .env file
load_dotenv()

def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    """Yield chat response content from the Groq API response."""
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

st.set_page_config(page_icon="💬", layout="wide", page_title="GitHub Chat App")

# Load configuration from config.yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Load Groq API key from environment variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY environment variable not found. Please set it in the .env file.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Define model details
model_option = "llama3-70b-8192"
model_info = config["models"][model_option]

# Display model information
st.sidebar.header("Model Information")
st.sidebar.markdown(f"**Name:** {model_info['name']}")
st.sidebar.markdown(f"**Developer:** {model_info['developer']}")
st.sidebar.markdown(f"**Description:** {model_info['description']}")

max_tokens_range = model_info["tokens"]

# Adjust max_tokens slider
max_tokens = st.sidebar.slider(
    "Max Tokens:",
    min_value=512,
    max_value=max_tokens_range,
    value=min(config["default_max_tokens"], max_tokens_range),
    step=512,
    help=f"Adjust the maximum number of tokens (words) for the model's response. Max for selected model: {max_tokens_range}"
)

# Utility functions
def parse_github_url(url):
    pattern = r"https://github\.com/([^/]+)/([^/]+)"
    match = re.match(pattern, url)
    return match.groups() if match else (None, None)

def clone_github_repo(repo_url):
    try:
        print('Cloning the repo ...')
        result = subprocess.run(["git", "clone", repo_url], check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to clone repository: {e}")
        return None

def validate_owner_repo(owner, repo):
    return bool(owner) and bool(repo)

# Setup a query engine
def setup_query_engine(github_url):
    owner, repo = parse_github_url(github_url)

    if validate_owner_repo(owner, repo):
        # Clone the GitHub repo & save it in a directory
        input_dir_path = f"/mount/src/chatwithgithubripo/{repo}"

        if os.path.exists(input_dir_path):
            pass
        else:
            clone_github_repo(github_url)

        # Load data from the cloned repository
        loader = SimpleDirectoryReader(
            input_dir=input_dir_path,
            required_exts=[".py", ".ipynb", ".js", ".ts", ".md"],
            recursive=True
        )

        try:
            docs = loader.load_data()

            # Create vector store and upload data
            Settings.embed_model = embed_model
            index = VectorStoreIndex.from_documents(docs, show_progress=True)

            # Setup a query engine
            Settings.llm = llm
            query_engine = index.as_query_engine(similarity_top_k=4)

            # Customize prompt template
            qa_prompt_tmpl_str = (
                "Context information is below.\n"
                "---------------------\n"
                "{context_str}\n"
                "---------------------\n"
                "Given the context information above I want you to think step by step to answer the query in a crisp manner, incase case you don't know the answer say 'I don't know!'.\n"
                "Query: {query_str}\n"
                "Answer: "
            )
            qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

            query_engine.update_prompts(
                {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
            )

            if docs:
                print("Data loaded successfully!!")
                print("Ready to chat!!")
            else:
                print("No data found, check if the repository is not empty!")

            return query_engine

        except Exception as e:
            print(f"An error occurred: {e}")
    else:
        print('Invalid GitHub repo, try again!')
        return None

# Sidebar input for GitHub URL
github_url = st.sidebar.text_input("GitHub Repository URL")

# Button to load and process the GitHub repository
if st.sidebar.button("Load Repository"):
    query_engine = setup_query_engine(github_url)
    st.session_state.query_engine = query_engine

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    avatar = '🤖' if message["role"] == "assistant" else '👨‍💻'
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

if prompt := st.chat_input("Enter your prompt here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar='👨‍💻'):
        st.markdown(prompt)

    # Fetch response from Groq API
    try:
        chat_completion = client.chat.completions.create(
            model=model_option,
            messages=[
                {
                    "role": m["role"],
                    "content": m["content"]
                }
                for m in st.session_state.messages
            ],
            max_tokens=max_tokens,
            stream=True
        )

        # Use the generator function with st.write_stream
        with st.chat_message("assistant", avatar="🤖"):
            chat_responses_generator = generate_chat_responses(chat_completion)
            full_response = st.write_stream(chat_responses_generator)
    except Exception as e:
        st.error(f"Error: {e}", icon="🚨")

    # Append the full response to session_state.messages
    if isinstance(full_response, str):
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response})
    else:
        # Handle the case where full_response is not a string
        combined_response = "\n".join(str(item) for item in full_response)
        st.session_state.messages.append(
            {"role": "assistant", "content": combined_response})

# Add a clear chat button
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
