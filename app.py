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

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .summary-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-top: 20px;
    }
    .upload-box {
        border: 2px dashed #1f77b4;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""
    if 'model_name' not in st.session_state:
        st.session_state.model_name = "meta-llama/llama-3-8b-instruct"
    if 'summary' not in st.session_state:
        st.session_state.summary = ""

def summarize_with_openrouter(news_texts, api_key, model_name):
    """
    Summarize news articles using OpenRouter API
    """
    if not news_texts:
        return "No valid news content found in the uploaded file."
    
    # Combine all news texts
    combined_text = "\n\n".join(news_texts)
    
    # Truncate if too long (OpenRouter has token limits)
    if len(combined_text) > 12000:
        combined_text = combined_text[:12000] + "... [truncated]"
    
    prompt = f"""Please analyze the following news articles and create a comprehensive summary suitable for a client email. 
    The summary should be professional, concise, and highlight the key developments and trends across all articles.

    News Articles:
    {combined_text}

    Please provide a well-structured summary that:
    1. Highlights the main topics and themes
    2. Identifies any significant trends or patterns
    3. Mentions key entities (companies, people, locations)
    4. Is written in a professional tone suitable for client communication
    5. Is approximately 200-300 words

    Summary:"""
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.3,
        "max_tokens": 1000
    }
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload),
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        else:
            return f"Error: {response.status_code} - {response.text}"
    
    except Exception as e:
        return f"Error connecting to OpenRouter: {str(e)}"

def process_uploaded_file(uploaded_file):
    """
    Process the uploaded file and extract news content
    """
    news_texts = []
    
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            # Try to find text columns
            text_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in ['text', 'content', 'summary', 'article'])]
            
            if text_columns:
                for col in text_columns:
                    news_texts.extend(df[col].dropna().astype(str).tolist())
            else:
                # If no obvious text columns, use all columns
                for col in df.columns:
                    news_texts.extend(df[col].dropna().astype(str).tolist())
        
        elif uploaded_file.name.endswith('.txt'):
            content = uploaded_file.getvalue().decode('utf-8')
            # Split by lines or paragraphs
            news_texts = [line.strip() for line in content.split('\n') if line.strip()]
        
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
            text_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in ['text', 'content', 'summary', 'article'])]
            
            if text_columns:
                for col in text_columns:
                    news_texts.extend(df[col].dropna().astype(str).tolist())
            else:
                for col in df.columns:
                    news_texts.extend(df[col].dropna().astype(str).tolist())
        
        else:
            return None, "Unsupported file format. Please upload CSV, TXT, or Excel files."
    
    except Exception as e:
        return None, f"Error processing file: {str(e)}"
    
    # Filter out very short texts (likely not news content)
    news_texts = [text for text in news_texts if len(text.strip()) > 20]
    
    return news_texts, None

def main():
    initialize_session_state()
    
    st.markdown('<h1 class="main-header">📰 News Summary Generator</h1>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.session_state.api_key = st.text_input(
            "OpenRouter API Key",
            type="password",
            help="Get your API key from https://openrouter.ai/",
            value=st.session_state.api_key
        )
        
        st.session_state.model_name = st.selectbox(
            "Choose Model",
            [
                "meta-llama/llama-3-8b-instruct",
                "google/gemma-7b-it",
                "mistralai/mistral-7b-instruct",
                "huggingfaceh4/zephyr-7b-beta",
                "openchat/openchat-7b"
            ],
            index=0,
            help="Select a free model from OpenRouter"
        )
        
        st.info("💡 Free models may have rate limits. For production use, consider premium models.")

    # Main content area
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    st.subheader("📤 Upload News File")
    uploaded_file = st.file_uploader(
        "Choose a file containing news summaries",
        type=['csv', 'txt', 'xlsx', 'xls'],
        help="Supported formats: CSV, TXT, Excel"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Process the file
        with st.spinner("Processing your file..."):
            news_texts, error = process_uploaded_file(uploaded_file)
        
        if error:
            st.error(error)
        elif not news_texts:
            st.warning("No valid news content found in the uploaded file.")
        else:
            st.success(f"✅ Found {len(news_texts)} news items in the file")
            
            # Show preview
            with st.expander("📋 Preview News Content"):
                for i, text in enumerate(news_texts[:5], 1):
                    st.write(f"**Item {i}:** {text[:200]}..." if len(text) > 200 else f"**Item {i}:** {text}")
            
            # Generate summary
            if st.button("🚀 Generate Summary", type="primary"):
                if not st.session_state.api_key:
                    st.error("Please enter your OpenRouter API key in the sidebar.")
                else:
                    with st.spinner("Generating comprehensive summary..."):
                        summary = summarize_with_openrouter(
                            news_texts, 
                            st.session_state.api_key, 
                            st.session_state.model_name
                        )
                        st.session_state.summary = summary
            
            # Display summary
            if st.session_state.summary:
                st.markdown('<div class="summary-box">', unsafe_allow_html=True)
                st.subheader("📝 Client Summary")
                st.write(st.session_state.summary)
                
                # Copy to clipboard button
                st.code(st.session_state.summary, language='markdown')
                
                st.download_button(
                    label="📥 Download Summary",
                    data=st.session_state.summary,
                    file_name="news_summary.txt",
                    mime="text/plain"
                )
                st.markdown('</div>', unsafe_allow_html=True)

    # Instructions section
    with st.expander("ℹ️ How to use this tool"):
        st.markdown("""
        1. **Get OpenRouter API Key**: 
           - Visit https://openrouter.ai/
           - Sign up and get your free API key
           - Enter it in the sidebar
        
        2. **Prepare Your File**:
           - CSV/Excel: Should contain columns with news text/content
           - TXT: Each line or paragraph should be a news summary
        
        3. **Upload & Generate**:
           - Upload your file
           - Click 'Generate Summary'
           - Copy or download the result for your client email
        
        **Note**: The app processes your data locally and only sends content to OpenRouter for summarization.
        """)

if __name__ == "__main__":
    main()