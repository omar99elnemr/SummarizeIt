# SummarizeIt ⚡🦜

SummarizeIt is a web application that utilizes Groq AI to summarize content from YouTube videos and websites. Created with Streamlit and LangChain, this tool allows users to quickly and easily obtain summaries of lengthy content.

## Features

- **Summarize YouTube Videos**: Extracts and summarizes transcripts from YouTube videos.
- **Summarize Web Pages**: Extracts and summarizes content from web pages.
- **Retry Logic**: Handles transient errors gracefully with retry mechanisms.
- **User-Friendly Interface**: Simple and intuitive interface built with Streamlit.

## Technologies Used

- **[Streamlit](https://streamlit.io/)**: For the web interface
- **[LangChain](https://langchain.readthedocs.io/)**: For managing the summarization chains
- **[Groq AI](https://groq.com/)**: For powerful language model capabilities
- **[YouTube Transcript API](https://pypi.org/project/youtube-transcript-api/)**: For extracting transcripts from YouTube videos
- **[BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/)**: For web scraping

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/omar99elnemr/SummarizeIt.git

2. Navigate to the project directory:
   ```bash
   cd SummarizeIt

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

4. Set up your Groq API key in Streamlit secrets. Create a ```.streamlit/secrets.toml``` file and add:
   ```TOML
   GROQ_API_KEY = "your_groq_api_key"
   
## Usage
1. Run the Streamlit app:
   ```bash
   streamlit run app.py
2. Open your web browser and go to http://localhost:8501.


### Note:
- For YouTube videos, the video must have subtitles/captions enabled.
- For websites, the content must be publicly accessible.
### Examples
Summarizing a YouTube Video
Summarizing a Web Page

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
