import streamlit as st
from transformers import pipeline

# Load the Hugging Face question-answering pipeline using DistilBERT
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad", clean_up_tokenization_spaces=True)

# Set up Streamlit app with tabs for navigation
tabs = st.tabs(["Overview", "Chat with QA Bot"])

# Overview tab
with tabs[0]:
    st.title("Welcome to the QA Chatbot App 🤖")
    st.write("""
        This application allows you to interact with a question-answering chatbot. 
        You can enter a context or passage, and then ask questions about it.
        
        ### Features
        - **DistilBERT-based QA**: Uses the DistilBERT model to answer questions.
        - **Interactive Chat**: Engage in a conversation with the QA bot to get answers from a given context.
        
        ### Key Applications
        #### Anti-Money Laundering (AML)
        - **Speeding up KYC Analysis**: In AML, analysts often need to review large volumes of customer information and documents during the Know Your Customer (KYC) process. This QA tool can help quickly extract key information from these documents, speeding up the review and decision-making process.
        - **Identifying Red Flags**: QA models can assist in identifying specific patterns or red flags in customer behavior based on large datasets, allowing analysts to focus on high-risk cases.

        #### Digital Internal Audit
        - **Efficient Documentation Review**: Internal audits often require the review of extensive documentation, policies, and procedures. This QA tool can significantly reduce the time required to extract relevant information from these documents, ensuring that auditors can focus on analysis rather than manual document searches.
        - **Improved Audit Insights**: By asking targeted questions, auditors can use the QA tool to quickly gain insights into compliance issues, control weaknesses, and operational risks, allowing for more efficient and effective audits.

        ### How to Use
        1. Enter a context for the bot to understand.
        2. Type in your questions, and the bot will provide the answers.
    """)

# Chat tab
with tabs[1]:
    st.title("Chat with QA Bot 🤖")
    st.write("This app allows you to ask questions based on a given context.")

    # Provide a context for questions
    context = st.text_area("Enter the context here:", height=150)

    # Create a chat interface
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Capture user question
    user_question = st.chat_input("Ask a question based on the above context:")

    # Process and display the chat if there's input
    if user_question and context:
        # Get the answer from the QA pipeline
        result = qa_pipeline(question=user_question, context=context)
        answer = result['answer']

        # Update chat history
        st.session_state.chat_history.append(("User", user_question))
        st.session_state.chat_history.append(("AI 🤖", answer))

    # Display chat history with icons
    for role, message in st.session_state.chat_history:
        icon = "🙂" if role == "User" else "🤖"
        st.chat_message(role).write(f"{icon} {message}")
