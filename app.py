import os
import validators
import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain_core.documents import Document

# Get API key from Streamlit secrets
if 'GROQ_API_KEY' not in st.secrets:
    st.error('GROQ_API_KEY is not set in the secrets. Please set it up in your Streamlit dashboard.')
    st.stop()

# Streamlit app configuration
st.set_page_config(
    page_title="Summarize Text From YT or Website",
    page_icon="⚡",
    layout="wide"
)

# Apply custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Summarize Text From YT or Website ⚡🦜")
st.subheader('Summarize URL')

# URL input
generic_url = st.text_input("Enter URL (YouTube video or website):", key="url_input")

# Initialize Groq LLM
@st.cache_resource
def get_llm():
    return ChatGroq(
        model="gemma-7b-it",  # Updated to use the correct model name
        groq_api_key=st.secrets['GROQ_API_KEY']
    )

llm = get_llm()

# Define summarization prompt
prompt_template = """
Provide a summary of the following content in 300 words:
Content: {text}

Please include:
- Main points and key takeaways
- Important details and context
- Conclusion or final thoughts
"""
prompt = PromptTemplate.from_template(prompt_template)

def extract_content(url):
    """Extract content from URL (YouTube or website)"""
    try:
        if "youtube.com" in url or "youtu.be" in url:
            # Use YoutubeLoader for YouTube videos
            loader = YoutubeLoader.from_youtube_url(
                url,
                add_video_info=True
            )
            docs = loader.load()
        else:
            # Use UnstructuredURLLoader for websites
            loader = UnstructuredURLLoader(
                urls=[url],
                ssl_verify=False,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                }
            )
            docs = loader.load()
        
        return docs if docs else None
    
    except Exception as e:
        st.error(f"Error extracting content: {str(e)}")
        return None

# Main summarization function
def summarize_content(url):
    try:
        # Extract content from URL
        docs = extract_content(url)
        if not docs:
            return None, "Could not extract content from the URL."
        
        # Create the LLM chain
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        
        # Create the StuffDocumentsChain
        stuff_chain = StuffDocumentsChain(
            llm_chain=llm_chain,
            document_variable_name="text"
        )
        
        # Run the chain
        summary = stuff_chain.run(docs)
        return summary, None
        
    except Exception as e:
        return None, f"Error processing content: {str(e)}"

# Button to trigger summarization
if st.button("Summarize Content"):
    if not generic_url.strip():
        st.error("Please provide a URL to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL (YouTube video or website)")
    else:
        with st.spinner("Processing... This may take a minute..."):
            summary, error = summarize_content(generic_url)
            
            if error:
                st.error(error)
            else:
                st.success("Summary generated successfully!")
                st.write(summary)

# Add footer with information
st.markdown("---")
st.markdown("### How to use:")
st.markdown("""
1. Enter a valid URL (YouTube video or website)
2. Click 'Summarize Content'
3. Wait for the summary to be generated

**Note**: 
- For YouTube videos, the video must be publicly accessible
- For websites, the content must be publicly accessible
""")

# Add GitHub link
st.sidebar.markdown("### About")
st.sidebar.markdown(
    "This app uses Groq AI to summarize content from YouTube videos and websites. "
    "Created with Streamlit and LangChain."
)
st.sidebar.markdown(
    "[View on GitHub](https://github.com/omar99elnemr/SummarizeIt)"
)
