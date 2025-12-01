from os import getenv

import boto3
import streamlit as st
from langchain_aws import BedrockLLM
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# AWS Bedrock Configuration
def get_bedrock_client():
    """Create and return a Bedrock runtime client."""
    return boto3.client(
        "bedrock-runtime", region_name=getenv("AWS_REGION", "us-west-2")
    )


def get_bedrock_llm():
    """Initialize Bedrock LLM."""
    bedrock_client = get_bedrock_client()
    model_id = getenv("BEDROCK_MODEL_ID", "anthropic.claude-v2")

    llm = BedrockLLM(
        client=bedrock_client, model_id=model_id, model_kwargs={"max_tokens": 512}
    )
    return llm


def main():
    """Main Streamlit application for chat interface."""
    st.title("🤖 AWS Bedrock Chat")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("What would you like to chat about?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get LLM response
        with st.chat_message("assistant"):
            try:
                # Use Bedrock LLM for response
                llm = get_bedrock_llm()
                response = llm.invoke(prompt)
                st.markdown(response)

                # Add assistant response to chat history
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
            except Exception as e:
                st.error(f"Error generating response: {e}")


if __name__ == "__main__":
    main()
