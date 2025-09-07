import streamlit as st
import pandas as pd
import requests
import json
import re

# Page configuration
st.set_page_config(
    page_title="News Summarizer",
    page_icon="📰",
    layout="wide"
)

def initialize_session_state():
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""
    if 'model_name' not in st.session_state:
        st.session_state.model_name = "meta-llama/llama-3-8b-instruct"
    if 'summary' not in st.session_state:
        st.session_state.summary = ""

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text.strip())
    return text

def process_file(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            articles = []
            for col in df.columns:
                if df[col].dtype == 'object':
                    articles.extend(df[col].dropna().astype(str).tolist())
            return [clean_text(article) for article in articles if len(clean_text(article)) > 20], None
            
        elif uploaded_file.name.endswith('.txt'):
            content = uploaded_file.getvalue().decode('utf-8')
            articles = [clean_text(article) for article in content.split('\n\n') if clean_text(article)]
            return [article for article in articles if len(article) > 20], None
            
        else:
            return None, "Unsupported file format"
            
    except Exception as e:
        return None, f"Error: {str(e)}"

def summarize_articles(articles, api_key, model_name):
    if not articles or not api_key:
        return "Please provide valid articles and API key"
    
    combined_text = "\n\n".join(articles[:20])  # Limit to first 20 articles
    
    prompt = f"""Please create a comprehensive summary of these news articles for a client email:

{combined_text}

Summary:"""
    
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 500
        }
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content'].strip()
        else:
            return f"API error: {response.status_code}"
            
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    initialize_session_state()
    
    st.title("📰 News Summary Generator")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Try to get API key from secrets
        try:
            if 'OPENROUTER_API_KEY' in st.secrets:
                default_key = st.secrets['OPENROUTER_API_KEY']
            else:
                default_key = ""
        except:
            default_key = ""
        
        st.session_state.api_key = st.text_input(
            "OpenRouter API Key",
            type="password",
            value=default_key or st.session_state.api_key
        )
        
        st.session_state.model_name = st.selectbox(
            "Model",
            ["meta-llama/llama-3-8b-instruct", "google/gemma-7b-it"],
            index=0
        )
    
    # Main area
    uploaded_file = st.file_uploader(
        "Upload news file (CSV or TXT)",
        type=['csv', 'txt']
    )
    
    if uploaded_file:
        articles, error = process_file(uploaded_file)
        
        if error:
            st.error(error)
        elif articles:
            st.info(f"Found {len(articles)} articles")
            
            if st.button("Generate Summary"):
                if not st.session_state.api_key:
                    st.error("Please enter API key")
                else:
                    with st.spinner("Creating summary..."):
                        summary = summarize_articles(articles, st.session_state.api_key, st.session_state.model_name)
                        st.session_state.summary = summary
            
            if st.session_state.summary:
                st.subheader("Summary")
                st.write(st.session_state.summary)
                
                st.download_button(
                    "Download Summary",
                    st.session_state.summary,
                    "summary.txt"
                )
        else:
            st.warning("No articles found in file")

if __name__ == "__main__":
    main()
