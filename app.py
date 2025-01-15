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
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
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

# Debug mode toggle
debug_mode = st.sidebar.checkbox("Debug Mode", value=False)

def get_youtube_id(url):
    """Extract YouTube video ID from URL"""
    try:
        parsed_url = urlparse(url)
        if parsed_url.hostname in ('youtu.be', 'www.youtu.be'):
            return parsed_url.path[1:]
        if parsed_url.hostname in ('youtube.com', 'www.youtube.com'):
            if parsed_url.path == '/watch':
                return parse_qs(parsed_url.query)['v'][0]
        return None
    except Exception:
        return None

def get_youtube_transcript(video_id):
    """Get YouTube transcript using youtube_transcript_api"""
    try:
        if debug_mode:
            st.write(f"Attempting to get transcript for video ID: {video_id}")
        
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return ' '.join(entry['text'] for entry in transcript)
    except Exception as e:
        if debug_mode:
            st.write(f"Error getting transcript: {str(e)}")
        return None

def extract_content(url):
    """Extract content from URL (YouTube or website) with fallback mechanisms"""
    try:
        if "youtube.com" in url or "youtu.be" in url:
            video_id = get_youtube_id(url)
            if not video_id:
                raise ValueError("Could not extract YouTube video ID")

            if debug_mode:
                st.write(f"Extracted video ID: {video_id}")

            # Try youtube_transcript_api first
            transcript = get_youtube_transcript(video_id)
            
            if transcript:
                if debug_mode:
                    st.write("Successfully got transcript using youtube_transcript_api")
                return [Document(page_content=transcript)]
            
            # Fallback to YoutubeLoader
            if debug_mode:
                st.write("Trying YoutubeLoader as fallback...")
            
            try:
                loader = YoutubeLoader.from_youtube_url(
                    url,
                    add_video_info=True,
                    language=['en']
                )
                docs = loader.load()
                if docs:
                    if debug_mode:
                        st.write("Successfully got content using YoutubeLoader")
                    return docs
            except Exception as e:
                if debug_mode:
                    st.write(f"YoutubeLoader failed: {str(e)}")
                raise Exception("Could not extract video content using any available method")
        else:
            # Handle website URLs
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
        if debug_mode:
            st.write(f"Final error in extract_content: {str(e)}")
        raise Exception(f"Error extracting content: {str(e)}")

# Initialize Groq LLM with error handling
@st.cache_resource
def get_llm():
    try:
        return ChatGroq(
            model="gemma2-9b-it",
            groq_api_key=st.secrets['GROQ_API_KEY']
        )
    except Exception as e:
        st.error(f"Error initializing Groq LLM: {str(e)}")
        st.error("Please check if the model name 'gemma2-9b-it' is correct and your API key is valid.")
        raise e

# Try to initialize the LLM
try:
    llm = get_llm()
except Exception:
    st.error("Failed to initialize the LLM. Please check the configuration.")
    st.stop()

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

# URL input
generic_url = st.text_input("Enter URL (YouTube video or website):", key="url_input")

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
