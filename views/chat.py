import streamlit as st
import uuid
import requests
import os
import time
from dotenv import load_dotenv

load_dotenv()

AGENT_SERVICE = os.getenv("AGENT_SERVICE", "localhost")
AGENT_PORT = int(os.getenv("AGENT_PORT"))

def word_stream(text: str):
    for world in text.split():
        yield world + " "
        time.sleep(0.05)

def main():
    st.set_page_config("Personal Assistant")

    if "session" not in st.session_state:
        st.session_state.session = {}

    if "current_session_id" not in st.session_state:
        new_session_id = str(uuid.uuid4())
        st.session_state.current_session_id = new_session_id
        st.session_state.session[new_session_id] = {
            "title": "",
            "user_id": "",
            "messages": []
        }
    
    with st.sidebar:
        if st.button("New chat", use_container_width=True, type="primary", key="new_chat_btn"):
            new_session_id = str(uuid.uuid4())
            st.session_state.current_session_id = new_session_id
            st.session_state.session[new_session_id] = {
                "title": "",
                "user_id": "",
                "messages": []
            }
            st.rerun()
        
        st.markdown("---")
        st.markdown("Your chats")
        
        for session_id in reversed(list(st.session_state.session.keys())):
            session = st.session_state.session[session_id]
            col_1, col_2 = st.columns([0.8, 0.2])
            with col_1:
                if st.button(session["title"] or "New chat", key=session_id, use_container_width=True):
                    st.session_state.current_session_id = session_id
                    st.rerun()
            with col_2:
                if st.button("X", key=session_id + "_delete_btn"):
                    del st.session_state.session[session_id]
                    if st.session_state.current_session_id == session_id:
                        st.session_state.current_session_id = None
                    st.rerun()
        
    st.title("Personal Assistant")
    
    session_id = st.session_state.current_session_id
    current_session = st.session_state.session[session_id]
    if current_session["user_id"]:
        st.info(f"User ID: {current_session['user_id']}")
    else:
        user_id = st.text_input("User ID", placeholder="Enter you user id")
    
    for msg in current_session["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    query = st.chat_input("ask anything...")
    if query:
        if not current_session["messages"]:
            current_session["user_id"] = user_id
            current_session["title"] = query[:50]
        current_session["messages"].append({
            "role": "human",
            "content": query
        })
        
        with st.chat_message("human"):
            st.markdown(query)
            
        with st.chat_message("assistant"):
            with st.spinner("thinking..."):
                response = requests.post(f"http://{AGENT_SERVICE}:{AGENT_PORT}/chat", data={
                    "query": query,
                    "session_id": session_id,
                    "user_id": current_session["user_id"]
                })
                
            inference_time = response.json()["inference_time"]
            content = st.write_stream(word_stream(response.json()["answer"]))   
        
        current_session["messages"].append({
            "role": "assistant",
            "content": content,
            "inference_time": inference_time
        })
        
        st.rerun()
                    
             
        
    
if __name__ == "__main__":
    main()