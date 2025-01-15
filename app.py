import os
import validators
import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
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
        model="gemma2-9b-it",  
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

def get_youtube_id(url):
    """Extract YouTube video ID from URL"""
    parsed_url = urlparse(url)
    if parsed_url.hostname in ('youtu.be', 'www.youtu.be'):
        return parsed_url.path[1:]
    if parsed_url.hostname in ('youtube.com', 'www.youtube.com'):
        if parsed_url.path == '/watch':
            return parse_qs(parsed_url.query).get('v', [None])[0]
    return None

def get_youtube_transcript(video_id):
    """Get YouTube video transcript with better error handling"""
    try:
        # Try getting the transcript directly first
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            return ' '.join([t['text'] for t in transcript])
        except Exception:
            # If direct method fails, try the list_transcripts approach
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # Try different approaches to get the transcript
            transcript = None
            
            # Try to get English transcript first
            try:
                transcript = transcript_list.find_transcript(['en'])
            except NoTranscriptFound:
                pass
            
            # If no English transcript, try to get any manually created transcript
            if not transcript:
                try:
                    transcript = transcript_list.find_manually_created_transcript()
                except NoTranscriptFound:
                    pass
            
            # If still no transcript, try to get auto-generated transcript
            if not transcript:
                try:
                    available_transcripts = list(transcript_list.transcripts.values())
                    if available_transcripts:
                        transcript = available_transcripts[0]
                except Exception:
                    pass
            
            # If we found a transcript
            if transcript:
                # Translate to English if needed
                if transcript.language_code != 'en':
                    try:
                        transcript = transcript.translate('en')
                    except Exception as e:
                        st.warning(f"Could not translate transcript to English: {str(e)}")
                
                transcript_text = ' '.join([t['text'] for t in transcript.fetch()])
                return transcript_text
            
            raise Exception("No transcript found after trying all methods")
            
    except Exception as e:
        detailed_error = f"""Could not extract transcript: {str(e)}
        
        This might be due to:
        1. The video doesn't have any captions/subtitles
        2. The captions are disabled or private
        3. YouTube API access issues
        
        Please try another video or verify that captions are available."""
        
        st.error(detailed_error)
        return None

def extract_text_from_url(url):
    """Extract text content from a website URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, verify=False)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'header', 'footer', 'nav']):
            element.decompose()
        
        # Extract text from main content areas
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content') or soup.body
        
        if main_content:
            paragraphs = main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            text = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
        else:
            # Fallback to all paragraphs if no main content area found
            paragraphs = soup.find_all('p')
            text = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
        
        return text if text.strip() else None
    except Exception as e:
        st.error(f"Web extraction error: {str(e)}")
        return None

# Main summarization function
def summarize_content(url):
    try:
        if "youtube.com" in url or "youtu.be" in url:
            video_id = get_youtube_id(url)
            if not video_id:
                return None, "Could not extract YouTube video ID from URL"
            
            transcript = get_youtube_transcript(video_id)
            if not transcript:
                return None, "Could not extract transcript from YouTube video. The video might not have subtitles enabled."
            
            docs = [Document(page_content=transcript)]
            
        else:
            # Extract text from website
            text = extract_text_from_url(url)
            if not text:
                return None, "No content could be extracted from the URL. The page might be protected or require authentication."
            
            docs = [Document(page_content=text)]
        
        if not docs or not docs[0].page_content.strip():
            return None, "No content could be extracted from the URL"
        
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

# Add GitHub link
st.sidebar.markdown("### About")
st.sidebar.markdown(
    "This app uses Groq AI to summarize content from YouTube videos and websites. "
    "Created with Streamlit and LangChain."
)
st.sidebar.markdown(
    "[View on GitHub](https://github.com/omar99elnemr/SummarizeIt)"
)
