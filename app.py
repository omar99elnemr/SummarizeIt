import os
import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

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
        model="gemma-7b-it",
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
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# Main summarization function
def summarize_content(url):
    try:
        if "youtube.com" in url or "youtu.be" in url:
            loader = YoutubeLoader.from_youtube_url(
                url,
                add_video_info=True
            )
        else:
            loader = UnstructuredURLLoader(
                urls=[url],
                ssl_verify=False,
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
                }
            )
        
        docs = loader.load()
        if not docs:
            return None, "No content could be extracted from the URL"
        
        chain = load_summarize_chain(
            llm,
            chain_type="stuff",
            prompt=prompt
        )
        
        return chain.run(docs), None
        
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
