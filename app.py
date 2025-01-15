import os
import validators
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_community.document_loaders import YoutubeLoader, WebBaseLoader
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs

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
        model="google/gemma-2-9b-it",
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
            return parse_qs(parsed_url.query)['v'][0]
    return None

def get_youtube_transcript(video_id):
    """Get YouTube video transcript"""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = ' '.join([t['text'] for t in transcript_list])
        return transcript
    except Exception as e:
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
                return None, "Could not extract transcript from YouTube video"
            
            # Create document for processing
            from langchain_core.documents import Document
            doc = Document(page_content=transcript)
            docs = [doc]
            
        else:
            # Use WebBaseLoader for non-YouTube URLs
            loader = WebBaseLoader(
                web_paths=[url],
                bs_kwargs=dict(
                    parse_only=None,
                    features="lxml"
                )
            )
            docs = loader.load()
        
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
                st.success("Summary generated successfully!")
                st.write(summary)

# Add footer with information
st.markdown("---")
st.markdown("### How to use:")
st.markdown("""
1. Enter a valid URL (YouTube video or website)
2. Click 'Summarize Content'
3. Wait for the summary to be generated

**Note**: Processing time may vary depending on the content length.
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
