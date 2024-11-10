"""
FileSight AI - Enhanced Query Chat Assistant
Description:
    This is an RAG application. User query in the code is 
Features:
    - Query Enhancement: Uses llm to improve the specificity and relevance to the context of user queries before it turns to final respond.
    - Document Handling: Dynamically loads and unloads documents to provide responses based on them.
    - Interactive Chat: Offers a seamless chat interface.
    - Gradio Interface
Version:
    0.2
Date:
    11-10-2024
"""

import os
import gradio as gr
import json
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Document,Settings, PromptTemplate, get_response_synthesizer
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.llms.nvidia import NVIDIA
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.vector_stores import SimpleVectorStore

# Global variables for the index and query engine
index = None
query_engine = None
chat_history_file = "chat_history.json"

# Function to get file paths from Gradio file objects
def get_files_from_input(file_objs):
    if not file_objs:
        return []
    return [file_obj.name for file_obj in file_objs]

# Function to load chat history
def load_chat_history():
    if not os.path.exists(chat_history_file):
        return []
    with open(chat_history_file, 'r') as file:
        return json.load(file)

# Function to save chat history
def save_chat_history(chat_history):
    with open(chat_history_file, 'w') as file:
        json.dump(chat_history, file)

# Function to initialize settings with API key and model
def initialize_settings(api_key, model_name):
    os.environ["NVIDIA_API_KEY"] = api_key

    if not os.getenv('NVIDIA_API_KEY'):
        raise ValueError("NVIDIA_API_KEY environment variable is not set")

    Settings.text_splitter = SentenceSplitter(chunk_size=500)

    # Initialize LLM
    try:
        Settings.llm = NVIDIA(
            model=model_name
        )
    except TypeError as e:
        print(f"Initialization error in NVIDIA LLM: {e}")
    
    # Initialize Embedding Model
    try:
        Settings.embed_model = NVIDIAEmbedding(
            model="NV-Embed-QA",
            truncate="END"
        )
    except TypeError as e:
        print(f"Initialization error in NVIDIA Embedding: {e}")

# Function to initialize index with optional documents
def initialize_index(documents=None):
    global index, query_engine
    if documents is None:
        documents = []  # You can add default documents here if desired
    vector_store = SimpleVectorStore()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    query_engine = index.as_query_engine(similarity_top_k=5, streaming=True)

# Function to unload documents
def unload_documents():
    global index, query_engine
    index = None
    query_engine = None
    return "Documents unloaded successfully."

# Function to load documents (used in Gradio UI)
def load_documents(file_objs):
    global index, query_engine
    try:
        if not file_objs:
            return "Error: No files selected."

        file_paths = get_files_from_input(file_objs)
        documents = []

        # Notify user that files are loading
        print("Loading files...")
        for file_path in file_paths:
            documents.extend(SimpleDirectoryReader(input_files=[file_path]).load_data())

        if not documents:
            return f"No documents found in the selected files."

        # Create a vector store and storage context
        vector_store = SimpleVectorStore()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Create the index from the documents
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

        # Create the query engine
        query_engine = index.as_query_engine(similarity_top_k=5, streaming=True)
        return f"Successfully loaded {len(documents)} documents from {len(file_paths)} files."
    except Exception as e:
        return f"Error loading documents: {str(e)}"

# Function to update visibility of custom model input
def update_custom_model_visibility(selected_model):
    if selected_model == "Custom":
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)

# Function to initialize the application with API key and model
def initialize_app(api_key, selected_model, custom_model_name):
    if selected_model == "Custom":
        model_name = custom_model_name
    else:
        model_name = selected_model

    try:
        initialize_settings(api_key, model_name)
        initialize_index()
        return "Initialization successful."
    except Exception as e:
        return f"Initialization failed: {str(e)}"

# Function to improve the user's query
def improve_query(user_message, recent_chat_history_str, relevant_chat_history_str):
    # Create a prompt to improve the query using f-strings
    improve_query_prompt = f"""\
Task:
Given the user's message and the recent conversation history, improve the user's query to make it more clear and detailed, considering the context.

Recent Chat History:
{recent_chat_history_str}

Relevant Chat History:
{relevant_chat_history_str}

User's Original Query:
{user_message}

Instruction: 
Provide only 1 sentence with only the improved query.
Do not include anything else other than the improved query.

Improved Query:"""

    # Use the LLM to generate the improved query
    print(improve_query_prompt)
    improved_query_response = query_engine.query(improve_query_prompt)
    # Ensure the response is a string
    if callable(improved_query_response):
        improved_query = improved_query_response().strip()
    else:
        improved_query = str(improved_query_response).strip()
    
    return improved_query

# Chat function
def chat_stream(user_message, chat_history):
    global index, query_engine
    if index is None or query_engine is None:
        return chat_history + [["Error", "No documents loaded. Please load documents first."]]
    if not user_message:
        return chat_history + [["Error", "No message provided."]]

    # Ensure chat_history is initialized
    if chat_history is None:
        chat_history = []

    # Append user message to chat history
    chat_history.append([user_message, None])
    save_chat_history(chat_history)

    try:
        # Process chat history to find relevant messages
        past_messages = chat_history[:-1]  # Exclude the current exchange

        # Convert past messages to documents (exclude entries without assistant response)
        past_documents = [Document(text=f"User: {msg[0]}\nAssistant: {msg[1]}") for msg in past_messages if msg[1]]

        # Create chat index and query engine
        chat_vector_store = SimpleVectorStore()
        chat_storage_context = StorageContext.from_defaults(vector_store=chat_vector_store)
        chat_index = VectorStoreIndex.from_documents(past_documents, storage_context=chat_storage_context)

        # Get the top T related messages
        T = 3
        retriever = VectorIndexRetriever(
            index=chat_index,
            similarity_top_k=T,
        )

        response_synthesizer = get_response_synthesizer()
        query_engine_chat = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
        )
        relevant_chat_history_response = query_engine_chat.query(user_message)

        # Extract relevant chat history
        relevant_chat_history_str = str(relevant_chat_history_response).strip()

        # Get the last K messages
        K = 5
        recent_messages = chat_history[-K:]
        recent_chat_history_str = "\n".join(
            [f"User: {msg[0]}\nAssistant: {msg[1]}" for msg in recent_messages if msg[1]]
        )

        # Improve the user's query
        improved_user_message = improve_query(user_message, recent_chat_history_str, relevant_chat_history_str)
        print(f"Improved Query: {improved_user_message}")

        # Get relevant documents using the improved query
        response = query_engine.query(improved_user_message)
        context_str = str(response)

        # Prepare the final prompt using f-strings
        qa_prompt = f"""\
Below is the context information.
{context_str}
------------------------------
Below is the chat_history reference.
{relevant_chat_history_str}
------------------------------
Below are the recent chats with the user.
{recent_chat_history_str}
------------------------------
Given the information and not prior knowledge, use the context, 
provide a helpful answer to the query.
Query: {improved_user_message}
Answer: """

        output_prompt_tmpl = PromptTemplate(qa_prompt)
        fmt_output_prompt = output_prompt_tmpl.format(
            recent_chat_history_str=recent_chat_history_str,
            relevant_chat_history_str=relevant_chat_history_str,
            improved_user_message=improved_user_message
        )
        # Generate response
        print(fmt_output_prompt)
        output_response = query_engine.query(fmt_output_prompt)
        chat_history[-1][1] = str(output_response)  # Update response
        save_chat_history(chat_history)

        return chat_history

    except Exception as e:
        error_message = f"Error: {str(e)}"
        chat_history[-1][1] = error_message  # Update response with error
        return chat_history

# Custom CSS for styling
custom_css = """
<style>
    body {
        background-color: #f5f5f5;
    }
    .gradio-container {
        max-width: 1000px;
        margin: auto;
    }
    .gr-button-primary {
        background-color: #4CAF50;
        color: white;
    }
    .gr-button-secondary {
        background-color: #f44336;
        color: white;
    }
    .gr-chatbot-message-user {
        background-color: #DCF8C6;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
    }
    .gr-chatbot-message-assistant {
        background-color: #FFFFFF;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
    }
</style>
"""

# Gradio interface
with gr.Blocks(css=custom_css, theme="default") as chatApp:
    gr.HTML(custom_css)  
    gr.Markdown("""
    <h1 style="text-align: center; color: #333333;"> FileSight AI</h1>
    <p style="text-align: center; color: #555555;">Enhance your document interactions with advanced AI-powered Q&A.</p>
    """)
    
    with gr.Tab("🔧 Settings"):
        gr.Markdown("## API Key and Model Selection")
        with gr.Row():
            api_key_input = gr.Textbox(
                label="Enter NVIDIA API Key 🔑",
                type="password",
                placeholder="Your NVIDIA API Key here",
                interactive=True
            )
        with gr.Row():
            model_options = [
                "meta/llama-3.1-8b-instruct",
                "meta/llama-3.2-3b-instruct",
                "nvidia/llama-3.1-nemotron-70b-instruct",
                "Custom"
            ]
            model_dropdown = gr.Dropdown(
                label="Select Model",
                choices=model_options,
                value="meta/llama-3.1-8b-instruct",
                interactive=True
            )
        with gr.Row():
            custom_model_input = gr.Textbox(
                label="Enter Custom Model Name",
                visible=False,
                placeholder="e.g., your-custom-model-name",
                interactive=True
            )
        with gr.Row():
            initialize_btn = gr.Button("🔄 Initialize", variant="primary")
            clear_history_btn = gr.Button("🧹 Clear Chat History", variant="secondary")
        with gr.Row():
            initialization_status = gr.Textbox(
                label="Initialization Status",
                interactive=False,
                placeholder="Status messages will appear here."
            )

        # Event handlers for model selection
        model_dropdown.change(
            update_custom_model_visibility,
            inputs=[model_dropdown],
            outputs=[custom_model_input]
        )

        # Initialize button
        initialize_btn.click(
            initialize_app,
            inputs=[api_key_input, model_dropdown, custom_model_input],
            outputs=[initialization_status]
        )

        # Clear chat history button
        def clear_chat_history():
            if os.path.exists(chat_history_file):
                os.remove(chat_history_file)
            return "Chat history cleared."

        clear_history_btn.click(
            clear_chat_history,
            inputs=[],
            outputs=[initialization_status]
        )

    with gr.Tab("📄 Documents"):
        gr.Markdown("## 📂 Document Management")
        with gr.Row():
            file_input = gr.File(
                label="Select Files to Load",
                file_count="multiple",
                interactive=True,
                type="filepath",  # Changed from "file" to "filepath"
                file_types=[".txt", ".pdf", ".docx"]  # Specify allowed file types
            )
            load_btn = gr.Button("📥 Load Documents", variant="primary")
            unload_btn = gr.Button("🗑️ Unload Documents", variant="secondary")
        with gr.Row():
            load_output = gr.Textbox(
                label="Load Status",
                interactive=False,
                placeholder="Status messages will appear here."
            )

        # Set up event handlers
        load_btn.click(load_documents, inputs=[file_input], outputs=[load_output])
        unload_btn.click(unload_documents, outputs=[load_output])

    with gr.Tab("💬 Chat"):
        gr.Markdown("## 💡 Chat Interface")
        with gr.Row():
            chatbot = gr.Chatbot(label="Conversation", height=500)
        with gr.Row():
            with gr.Column(scale=8):
                msg = gr.Textbox(
                    label="Enter your question",
                    placeholder="Type your message here...",
                    lines=1,
                    interactive=True
                )
            with gr.Column(scale=2):
                send_btn = gr.Button("📨 Send", variant="primary")
                clear_btn = gr.Button("🧹 Clear Chat", variant="secondary")
        
        # Set up event handlers
        send_btn.click(chat_stream, inputs=[msg, chatbot], outputs=[chatbot])
        send_btn.click(lambda: "", outputs=[msg])  # Clear input box after submission
        clear_btn.click(lambda: [], None, chatbot, queue=False)

    gr.Markdown("""
    ---
    <p style="text-align: center; color: #777777;">
        &copy; 2024 Your Company. All rights reserved.
    </p>
    """)

# Launch gradio interface
if __name__ == "__main__":
    chatApp.launch()
