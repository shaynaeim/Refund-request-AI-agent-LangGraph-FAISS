"""Streamlit Chatbot App for Refund Processing System."""

import streamlit as st
from datetime import datetime
import sys
import os
import yaml

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Load config file
config_path = "src/config.yaml"
if not os.path.exists(config_path):
    st.error(f"Configuration file '{config_path}' not found. Please create it with your model and credential settings.")
    st.stop()

# Load YAML config
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

os.environ["OPENAI_API_KEY"] = config['credentials']['openai']['api_key']

from src.agents import RefundProcessingSystem




def initialize_session_state():
    """Initialize Streamlit session state."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'refund_system' not in st.session_state:
        with st.spinner("Initializing multi-agent system..."):
            # Create a new instance with config
            st.session_state.refund_system = RefundProcessingSystem(config_path)


def add_message(role: str, content: str):
    """Add message to conversation."""
    timestamp = datetime.now().strftime("%I:%M %p")
    st.session_state.messages.append({
        "role": role,
        "content": content,
        "timestamp": timestamp
    })


def display_messages():
    """Display conversation messages."""
    for message in st.session_state.messages:
        timestamp = message.get("timestamp", "")
        
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
                if timestamp:
                    st.caption(f"You - {timestamp}")
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])
                if timestamp:
                    st.caption(f"Support Agent - {timestamp}")


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Customer Service Chat",
        page_icon="💬",
        layout="centered"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.title("💬 Customer Service Chat")
    st.subheader("Refund & Return Support")
    
    # Sidebar with system status
    with st.sidebar:
        st.header("System Status")
        if hasattr(st.session_state, 'refund_system'):
            st.success("Multi-Agent System Ready")
        else:
            st.error("System Not Initialized")
        
        # Session controls
        st.header("Session Controls")
        if st.button("Clear Conversation"):
            st.session_state.messages = []
            # Also clear the agent's conversation history
            if hasattr(st.session_state, 'refund_system') and hasattr(st.session_state.refund_system, 'conversation_history'):
                st.session_state.refund_system.conversation_history = []
            st.rerun()
            
      
    
    # Chat interface
    st.markdown("---")
    
    # Display conversation
    display_messages()
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message
        add_message("user", prompt)
        
        # Get response from supervisor agent
        with st.spinner("Processing your request..."):
            response = st.session_state.refund_system.chat_with_supervisor(prompt)
        
        # Add assistant response
        add_message("assistant", response)
        
        # Rerun to update display
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.caption("Powered by Multi-Agent LangGraph System | Customer Service Bot")


if __name__ == "__main__":
    main()