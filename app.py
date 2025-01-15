import streamlit as st
import validators
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube
from bs4 import BeautifulSoup
import requests
from langchain_core.documents import Document
import time

# Get API key from Streamlit secrets
if 'GROQ_API_KEY' not in st.secrets:
    st.error('GROQ_API_KEY is not set in the secrets. Please set it up in your Streamlit dashboard.')
    st.stop()

# Streamlit app configuration
st.set_page_config(page_title="Summarize Text From YT or Website", page_icon="⚡", layout="wide")

st.title("Summarize Text From YT or Website ⚡🦜")
st.subheader('Summarize URL')

# URL input
generic_url = st.text_input("Enter URL (YouTube video or website):", key="url_input")

# Initialize Groq LLM
@st.cache_resource
def get_llm():
    return ChatGroq(model="gemma2-9b-it", groq_api_key=st.secrets['GROQ_API_KEY'])

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
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

def get_youtube_transcript(video_url, retries=3):
    """Get YouTube video transcript using youtube_transcript_api with retry logic"""
    for attempt in range(retries):
        try:
            video_id = YouTube(video_url).video_id
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            transcript_text = ' '.join(entry['text'] for entry in transcript)
            return clean_transcript(transcript_text)
        except Exception:
            time.sleep(2)  # Wait before retrying
    return None

def clean_transcript(transcript_text):
    """Clean up the transcript text by removing repeated phrases and unnecessary characters"""
    lines = transcript_text.split('. ')
    cleaned_lines = []
    for line in lines:
        cleaned_line = ' '.join(dict.fromkeys(line.split()))
        cleaned_lines.append(cleaned_line)
    return '. '.join(cleaned_lines)

def extract_text_from_url(url, retries=3):
    """Extract text content from a website URL with retry logic"""
    for attempt in range(retries):
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            paragraphs = soup.find_all('p')
            text = ' '.join(p.get_text().strip() for p in paragraphs if p.get_text().strip())
            return [Document(page_content=text)] if text.strip() else None
        except Exception:
            time.sleep(2)  # Wait before retrying
    return None

def summarize_content(url):
    try:
        if "youtube.com" in url or "youtu.be" in url:
            transcript = get_youtube_transcript(url)
            if not transcript:
                return None, "Could not extract transcript from YouTube video."
            docs = [Document(page_content=transcript)]
        else:
            docs = extract_text_from_url(url)
            if not docs:
                return None, "No content could be extracted from the URL. The page might be protected or require authentication."
        
        chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
        summary = chain.run(docs)
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
                if "transcript" in error:
                    st.info("Failed to extract transcript from the YouTube video. Please try again.")
                else:
                    st.info("Failed to extract content from the URL. Please try again.")
            else:
                st.success(summary)

# Add footer with information
st.markdown("---")
st.markdown("### How to use:")
st.markdown("""
1. Enter a valid URL (YouTube video or website)
2. Click 'Summarize Content'
3. Wait for the summary to be generated

**Note**: 
- For YouTube videos, the video must have subtitles/captions enabled
- For websites, the content must be publicly accessible
""")
st.sidebar.markdown("### About")
st.sidebar.markdown(
    "This app uses Groq AI to summarize content from YouTube videos and websites. "
    "Created with Streamlit and LangChain."
)
st.sidebar.markdown(
    "[View on GitHub](https://github.com/omar99elnemr/SummarizeIt)"
)
