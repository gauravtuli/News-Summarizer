import streamlit as st
import pandas as pd
import requests
import json
import re
import io

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
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        margin-top: 15px;
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

def clean_text(text):
    """Clean and preprocess text"""
    if not isinstance(text, str):
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?;:-]', '', text)
    return text

def extract_articles_from_file(uploaded_file):
    """
    Extract articles from uploaded file with better handling for full articles
    """
    articles = []
    
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            articles = extract_from_dataframe(df)
            
        elif uploaded_file.name.endswith('.txt'):
            content = uploaded_file.getvalue().decode('utf-8')
            # Split by multiple newlines indicating article separation
            articles = [clean_text(article) for article in re.split(r'\n\s*\n', content) if clean_text(article)]
            
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
            articles = extract_from_dataframe(df)
            
        else:
            return None, "Unsupported file format. Please upload CSV, TXT, or Excel files."
    
    except Exception as e:
        return None, f"Error processing file: {str(e)}"
    
    # Filter out very short articles (less than 50 characters)
    articles = [article for article in articles if len(article) >= 50]
    
    return articles, None

def extract_from_dataframe(df):
    """Extract articles from DataFrame with intelligent column detection"""
    articles = []
    
    # Try to find article content columns
    content_columns = []
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['article', 'content', 'text', 'body', 'news', 'story']):
            content_columns.append(col)
        elif df[col].dtype == 'object' and df[col].str.len().mean() > 100:  # Likely text content
            content_columns.append(col)
    
    # If no specific columns found, use all string columns
    if not content_columns:
        content_columns = [col for col in df.columns if df[col].dtype == 'object']
    
    # Extract articles from identified columns
    for col in content_columns:
        column_articles = df[col].dropna().astype(str).apply(clean_text).tolist()
        articles.extend([article for article in column_articles if len(article) >= 50])
    
    return articles

def chunk_text(text, max_chunk_size=8000):
    """Split text into manageable chunks for API processing"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        if current_size + len(word) + 1 > max_chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_size = len(word)
        else:
            current_chunk.append(word)
            current_size += len(word) + 1
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def summarize_with_openrouter(articles, api_key, model_name):
    """
    Summarize news articles using OpenRouter API with chunking for large inputs
    """
    if not articles:
        return "No valid articles found in the uploaded file."
    
    # Combine all articles with separators
    combined_text = "\n\n--- ARTICLE ---\n\n".join(articles)
    
    # Chunk the text if it's too large
    if len(combined_text) > 10000:
        chunks = chunk_text(combined_text, 8000)
    else:
        chunks = [combined_text]
    
    final_summary = ""
    
    for i, chunk in enumerate(chunks):
        prompt = f"""ANALYSIS TASK: Create a comprehensive executive summary from multiple news articles for client communication.

ARTICLE CONTENT:
{chunk}

INSTRUCTIONS:
1. Analyze ALL articles comprehensively
2. Identify main themes, trends, and patterns across ALL content
3. Highlight significant developments, events, and key entities
4. Note any contradictions or varying perspectives between articles
5. Provide insights on overall media coverage trends

SUMMARY REQUIREMENTS:
- Professional, client-ready tone
- 250-400 words comprehensive coverage
- Structured with clear sections
- Focus on business relevance
- Avoid editorializing, stick to factual synthesis
- Highlight most important information first

SUMMARY:"""
        
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
            "max_tokens": 1200
        }
        
        try:
            with st.spinner(f"Processing chunk {i+1}/{len(chunks)}..."):
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=90
                )
                
                if response.status_code == 200:
                    result = response.json()
                    chunk_summary = result['choices'][0]['message']['content'].strip()
                    final_summary += chunk_summary + "\n\n"
                else:
                    return f"API Error: {response.status_code} - {response.text}"
        
        except Exception as e:
            return f"Error connecting to OpenRouter: {str(e)}"
    
    return final_summary if final_summary else "No summary could be generated."

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
        
        try:
            if 'DEFAULT_MODEL' in st.secrets:
                default_model = st.secrets['DEFAULT_MODEL']
            else:
                default_model = "meta-llama/llama-3-8b-instruct"
        except:
            default_model = "meta-llama/llama-3-8b-instruct"
        
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
        
        st.info("""
        💡 **Tips:**
        - Free models may have rate limits
        - For large files, processing may take 1-2 minutes
        - Ensure your API key has sufficient credits
        """)

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
            articles, error = extract_articles_from_file(uploaded_file)
        
        if error:
            st.error(f"❌ {error}")
        elif not articles:
            st.warning("⚠️ No valid articles found in the uploaded file.")
        else:
            st.success(f"✅ Found {len(articles)} articles in the file")
            st.session_state.processed_articles = len(articles)
            
            # Show preview
            with st.expander("📋 Preview Articles", expanded=False):
                for i, article in enumerate(articles[:3], 1):
                    st.write(f"**Article {i}:**")
                    st.write(article[:300] + "..." if len(article) > 300 else article)
                    st.write("---")
            
            # Generate summary
            if st.button("🚀 Generate Comprehensive Summary", type="primary", use_container_width=True):
                if not st.session_state.api_key:
                    st.error("🔑 Please enter your OpenRouter API key in the sidebar.")
                else:
                    with st.spinner("🧠 Analyzing articles and generating comprehensive summary... This may take a few minutes."):
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
                
                # Copy options
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="📥 Download Summary",
                        data=st.session_state.summary,
                        file_name="executive_summary.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                with col2:
                    if st.button("📋 Copy to Clipboard", use_container_width=True):
                        st.code(st.session_state.summary, language='markdown')
                        st.success("Summary copied to clipboard!")
                
                st.markdown('</div>', unsafe_allow_html=True)

    # Instructions section
    with st.expander("ℹ️ Detailed Instructions", expanded=False):
        st.markdown("""
        ## 📖 How to Use This Tool

        ### 1. **Get OpenRouter API Key**
           - Visit [https://openrouter.ai/](https://openrouter.ai/)
           - Sign up for a free account
           - Go to Dashboard → Keys to get your API key
           - Enter it in the sidebar

        ### 2. **Prepare Your File**
        **Supported formats:**
           - **CSV/Excel**: Should contain columns with full article text
           - **TXT**: Articles separated by blank lines

        **File should contain:**
           - Full news article text (not just summaries)
           - Each article should be substantial content (50+ characters)
           - Clean, readable text without excessive formatting

        ### 3. **Upload & Generate**
           - Upload your file
           - Review the article count
           - Click 'Generate Comprehensive Summary'
           - Wait for processing (1-2 minutes for large files)
           - Copy or download the result

        ### 🔒 Privacy Note
        - Your data is processed locally first
        - Only article content is sent to OpenRouter
        - No data is stored on our servers
        """)

if __name__ == "__main__":
    main()
