import streamlit as st
import pandas as pd
import requests
import json
import re

# Page configuration
st.set_page_config(
    page_title="News Summarizer Pro",
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
        padding: 25px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-top: 20px;
        line-height: 1.6;
    }
    .upload-box {
        border: 2px dashed #1f77b4;
        padding: 25px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 25px;
        background-color: #fafafa;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
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
    if 'processed_articles' not in st.session_state:
        st.session_state.processed_articles = 0
    if 'all_articles' not in st.session_state:
        st.session_state.all_articles = []

def clean_text(text):
    """Clean and preprocess text"""
    if not isinstance(text, str) or pd.isna(text):
        return ""
    
    # Remove extra whitespace and clean up text
    text = re.sub(r'\s+', ' ', text.strip())
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?;:-]', '', text)
    return text

def extract_all_articles_from_file(uploaded_file):
    """
    Extract ALL articles from uploaded file with comprehensive processing
    """
    all_articles = []
    
    try:
        if uploaded_file.name.endswith('.csv'):
            # Read CSV with proper encoding handling
            try:
                df = pd.read_csv(uploaded_file)
            except UnicodeDecodeError:
                df = pd.read_csv(uploaded_file, encoding='latin-1')
            
            # Process ALL string columns
            for col in df.columns:
                if df[col].dtype == 'object':  # String columns
                    articles = df[col].dropna().astype(str).apply(clean_text).tolist()
                    # Filter out short texts and add to main list
                    all_articles.extend([article for article in articles if len(article) > 30])
            
        elif uploaded_file.name.endswith('.txt'):
            # Read text file with proper encoding
            try:
                content = uploaded_file.getvalue().decode('utf-8')
            except UnicodeDecodeError:
                content = uploaded_file.getvalue().decode('latin-1')
            
            # Split by multiple newlines (article separators)
            articles = re.split(r'\n\s*\n', content)
            all_articles = [clean_text(article) for article in articles if clean_text(article)]
            
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            # Read Excel file
            df = pd.read_excel(uploaded_file)
            
            # Process ALL sheets and ALL string columns
            for col in df.columns:
                if df[col].dtype == 'object':  # String columns
                    articles = df[col].dropna().astype(str).apply(clean_text).tolist()
                    all_articles.extend([article for article in articles if len(article) > 30])
        
        else:
            return None, "Unsupported file format. Please upload CSV, TXT, or Excel files."
    
    except Exception as e:
        return None, f"Error processing file: {str(e)}"
    
    # Remove duplicates and empty articles
    all_articles = [article for article in all_articles if article and len(article) > 30]
    
    return all_articles, None

def chunk_articles_for_processing(articles, max_total_length=12000):
    """
    Split articles into manageable chunks for API processing
    """
    chunks = []
    current_chunk = []
    current_length = 0
    
    for article in articles:
        article_length = len(article)
        
        if current_length + article_length > max_total_length:
            # Save current chunk and start new one
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = [article]
            current_length = article_length
        else:
            current_chunk.append(article)
            current_length += article_length
    
    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def summarize_with_openrouter(articles, api_key, model_name):
    """
    Summarize ALL articles using OpenRouter API with proper chunking
    """
    if not articles:
        return "No valid articles found in the uploaded file."
    
    # Chunk articles for processing
    article_chunks = chunk_articles_for_processing(articles)
    st.info(f"Processing {len(article_chunks)} chunk(s) containing {len(articles)} total articles")
    
    all_summaries = []
    
    for chunk_index, chunk in enumerate(article_chunks):
        combined_text = "\n\n--- ARTICLE ---\n\n".join(chunk)
        
        prompt = f"""ANALYSIS TASK: Create a comprehensive executive summary from multiple news articles for client communication.

ARTICLE CONTENT:
{combined_text}

INSTRUCTIONS:
1. Analyze ALL articles comprehensively
2. Identify main themes, trends, and patterns across ALL content
3. Highlight significant developments, events, and key entities
4. Provide insights on overall media coverage trends
5. Create a professional, client-ready summary

SUMMARY REQUIREMENTS:
- Comprehensive coverage of all articles
- Professional business tone
- 300-500 words
- Structured with clear sections
- Focus on most important information

SUMMARY:"""
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 1500
        }
        
        try:
            with st.spinner(f"Processing chunk {chunk_index + 1}/{len(article_chunks)}..."):
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=120
                )
                
                if response.status_code == 200:
                    result = response.json()
                    chunk_summary = result['choices'][0]['message']['content'].strip()
                    all_summaries.append(chunk_summary)
                else:
                    return f"API Error: {response.status_code} - {response.text}"
        
        except Exception as e:
            return f"Error connecting to OpenRouter: {str(e)}"
    
    # Combine all chunk summaries
    if all_summaries:
        return "\n\n".join(all_summaries)
    else:
        return "No summary could be generated."

def main():
    initialize_session_state()
    
    st.markdown('<h1 class="main-header">📰 News Summary Generator Pro</h1>', unsafe_allow_html=True)
    st.markdown("### Transform multiple news articles into one comprehensive client summary")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Try to get API key from secrets first
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
            help="Get your free API key from https://openrouter.ai/",
            value=default_key or st.session_state.api_key
        )
        
        st.session_state.model_name = st.selectbox(
            "Choose Model",
            [
                "meta-llama/llama-3-8b-instruct",
                "google/gemma-7b-it",
                "mistralai/mistral-7b-instruct"
            ],
            index=0
        )

    # Main content area
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    st.subheader("📤 Upload News Articles File")
    uploaded_file = st.file_uploader(
        "Choose a file containing news articles",
        type=['csv', 'txt', 'xlsx', 'xls'],
        help="Supported formats: CSV, TXT, Excel. Files should contain full article text."
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Process the file
        with st.spinner("Analyzing your articles..."):
            articles, error = extract_all_articles_from_file(uploaded_file)
        
        if error:
            st.error(f"❌ {error}")
        elif not articles:
            st.warning("⚠️ No valid articles found in the uploaded file.")
        else:
            st.session_state.all_articles = articles
            st.success(f"✅ Found {len(articles)} articles in the file")
            
            # Show detailed preview
            with st.expander("📋 Article Details", expanded=True):
                st.write(f"**Total Articles Processed:** {len(articles)}")
                st.write(f"**Total Characters:** {sum(len(article) for article in articles):,}")
                
                # Show sample articles
                st.subheader("Sample Articles:")
                for i, article in enumerate(articles[:5], 1):
                    st.write(f"**Article {i}** ({len(article)} characters):")
                    st.write(article[:200] + "..." if len(article) > 200 else article)
                    st.write("---")
            
            # Generate summary
            if st.button("🚀 Generate Comprehensive Summary", type="primary"):
                if not st.session_state.api_key:
                    st.error("🔑 Please enter your OpenRouter API key in the sidebar.")
                else:
                    with st.spinner("🧠 Analyzing all articles and generating comprehensive summary... This may take a few minutes."):
                        summary = summarize_with_openrouter(
                            articles, 
                            st.session_state.api_key, 
                            st.session_state.model_name
                        )
                        st.session_state.summary = summary
            
            # Display summary
            if st.session_state.summary:
                st.markdown('<div class="summary-box">', unsafe_allow_html=True)
                st.subheader("📝 Executive Summary for Client")
                st.write(st.session_state.summary)
                
                # Download button
                st.download_button(
                    label="📥 Download Summary",
                    data=st.session_state.summary,
                    file_name="comprehensive_news_summary.txt",
                    mime="text/plain"
                )
                st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
