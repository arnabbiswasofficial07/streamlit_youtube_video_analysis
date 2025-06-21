import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
import warnings

warnings.filterwarnings('ignore')

# Try to import transformers for toxicity detection
try:
    from transformers import pipeline

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    st.warning("⚠️ Transformers not available. Toxicity detection will be disabled.")

# Page configuration
st.set_page_config(
    page_title="YouTube Video Analyzer",
    page_icon="📺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF0000;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">📺 YouTube Video Analyzer</h1>', unsafe_allow_html=True)

# Sidebar for API configuration
st.sidebar.header("🔧 Configuration")
api_key = st.sidebar.text_input("YouTube Data API v3 Key", type="password",
                                help="Get your API key from Google Cloud Console")

# Toxicity detection settings
st.sidebar.subheader("🚨 Toxicity Detection")
enable_toxicity = st.sidebar.checkbox("Enable Toxicity Detection", value=True,
                                      help="Disable if having issues with transformer models")
toxicity_method = st.sidebar.selectbox("Toxicity Method",
                                       ["Auto (Try Transformers)", "Keyword-based Only"],
                                       help="Auto tries transformer models first, falls back to keywords")

# Initialize session state
if 'video_data' not in st.session_state:
    st.session_state.video_data = None
if 'comments_data' not in st.session_state:
    st.session_state.comments_data = None


def extract_video_id(url):
    """Extract video ID from YouTube URL"""
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
        r'youtube\.com\/watch\?.*v=([^&\n?#]+)'
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def fetch_video_details(video_id, api_key):
    """Fetch video details using YouTube Data API v3"""
    try:
        url = f"https://www.googleapis.com/youtube/v3/videos"
        params = {
            'part': 'snippet,statistics',
            'id': video_id,
            'key': api_key
        }

        response = requests.get(url, params=params)
        data = response.json()

        if 'items' in data and len(data['items']) > 0:
            item = data['items'][0]
            snippet = item['snippet']
            statistics = item['statistics']

            return {
                'title': snippet.get('title', 'N/A'),
                'channel_name': snippet.get('channelTitle', 'N/A'),
                'published_date': snippet.get('publishedAt', 'N/A'),
                'description': snippet.get('description', 'N/A'),
                'view_count': int(statistics.get('viewCount', 0)),
                'like_count': int(statistics.get('likeCount', 0)),
                'comment_count': int(statistics.get('commentCount', 0)),
                'thumbnail': snippet.get('thumbnails', {}).get('high', {}).get('url', '')
            }
        else:
            st.error("Video not found or API error")
            return None

    except Exception as e:
        st.error(f"Error fetching video details: {str(e)}")
        return None


def fetch_comments(video_id, api_key, max_results=100):
    """Fetch comments using YouTube Data API v3"""
    try:
        url = f"https://www.googleapis.com/youtube/v3/commentThreads"
        params = {
            'part': 'snippet',
            'videoId': video_id,
            'key': api_key,
            'maxResults': min(max_results, 100),
            'order': 'relevance'
        }

        response = requests.get(url, params=params)
        data = response.json()

        comments = []
        if 'items' in data:
            for item in data['items']:
                comment = item['snippet']['topLevelComment']['snippet']
                comments.append({
                    'text': comment['textDisplay'],
                    'author': comment['authorDisplayName'],
                    'like_count': comment['likeCount'],
                    'published_at': comment['publishedAt']
                })

        return comments

    except Exception as e:
        st.error(f"Error fetching comments: {str(e)}")
        return []


def clean_text(text):
    """Clean and preprocess text"""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove extra whitespaces
    text = ' '.join(text.split())
    return text


def analyze_sentiment_textblob(text):
    """Analyze sentiment using TextBlob"""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    if polarity > 0.1:
        return 'Positive'
    elif polarity < -0.1:
        return 'Negative'
    else:
        return 'Neutral'


def analyze_sentiment_vader(text):
    """Analyze sentiment using VADER"""
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']

    if compound >= 0.05:
        return 'Positive'
    elif compound <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'


def detect_toxicity_simple(texts):
    """Simple keyword-based toxicity detection as fallback"""
    # Common toxic keywords (simplified list)
    toxic_keywords = [
        'hate', 'stupid', 'idiot', 'moron', 'dumb', 'kill', 'die', 'death',
        'racist', 'sexist', 'homophobic', 'transphobic', 'nazi', 'fascist',
        'terrorist', 'violence', 'murder', 'assault', 'abuse', 'harassment',
        'bully', 'threatening', 'disgusting', 'pathetic', 'loser', 'trash'
    ]

    results = []
    for text in texts:
        text_lower = text.lower()
        is_toxic = any(keyword in text_lower for keyword in toxic_keywords)
        results.append('Toxic' if is_toxic else 'Non-toxic')

    return results


def detect_toxicity(texts):
    """Detect toxicity using transformers with multiple fallback options"""
    if not TRANSFORMERS_AVAILABLE:
        return ['Unknown'] * len(texts)

    # List of models to try in order of preference
    models_to_try = [
        "unitary/toxic-bert",
        "cardiffnlp/twitter-roberta-base-hate-latest",
        "microsoft/DialoGPT-medium"  # Last resort
    ]

    for model_name in models_to_try:
        try:
            st.info(f"Trying toxicity model: {model_name}")

            if model_name == "cardiffnlp/twitter-roberta-base-hate-latest":
                # This model is specifically for hate speech detection
                classifier = pipeline("text-classification",
                                      model=model_name,
                                      device=-1,
                                      truncation=True,
                                      max_length=512)
            else:
                # Standard toxicity detection
                classifier = pipeline("text-classification",
                                      model=model_name,
                                      device=-1,
                                      truncation=True,
                                      max_length=512)

            results = []
            batch_size = 5  # Smaller batch size for stability

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]

                try:
                    # Process batch
                    batch_results = classifier(batch_texts)

                    for result in batch_results:
                        # Handle different result formats
                        if isinstance(result, list):
                            result = result[0]

                        label = result['label'].upper()
                        score = result['score']

                        # Different models use different labels
                        toxic_labels = ['TOXIC', 'HATE', 'OFFENSIVE', 'HARMFUL']

                        if any(toxic_label in label for toxic_label in toxic_labels) and score > 0.6:
                            results.append('Toxic')
                        else:
                            results.append('Non-toxic')

                except Exception as batch_error:
                    st.warning(f"Batch error with {model_name}: {str(batch_error)}")
                    results.extend(['Unknown'] * len(batch_texts))

            st.success(f"Successfully used model: {model_name}")
            return results

        except Exception as model_error:
            st.warning(f"Model {model_name} failed: {str(model_error)}")
            continue

    # If all models fail, use simple keyword-based detection
    st.info("All transformer models failed. Using keyword-based toxicity detection...")
    return detect_toxicity_simple(texts)


def create_sentiment_chart(sentiment_counts):
    """Create sentiment distribution chart"""
    fig = px.pie(
        values=list(sentiment_counts.values()),
        names=list(sentiment_counts.keys()),
        title="Sentiment Distribution",
        color_discrete_map={
            'Positive': '#00FF00',
            'Negative': '#FF0000',
            'Neutral': '#808080'
        }
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig


def create_wordcloud(texts):
    """Create word cloud from texts"""
    try:
        # Combine all texts
        combined_text = ' '.join([clean_text(text) for text in texts])

        # Create word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='viridis',
            max_words=100,
            relative_scaling=0.5
        ).generate(combined_text)

        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        plt.title('Most Common Words in Comments', fontsize=16, pad=20)

        return fig
    except Exception as e:
        st.error(f"Error creating word cloud: {str(e)}")
        return None


# Main app interface
if not api_key:
    st.warning("⚠️ Please enter your YouTube Data API v3 key in the sidebar to continue.")
    st.markdown("""
    ### How to get YouTube Data API v3 key:
    1. Go to [Google Cloud Console](https://console.cloud.google.com/)
    2. Create a new project or select existing one
    3. Enable YouTube Data API v3
    4. Create credentials (API key)
    5. Copy and paste the API key in the sidebar
    """)
else:
    # Input section
    st.header("🎥 Video Input")
    video_url = st.text_input("Enter YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=dQw4w9WgXcQ")

    col1, col2 = st.columns([1, 3])
    with col1:
        analyze_button = st.button("🔍 Analyze Video", type="primary")
    with col2:
        max_comments = st.slider("Maximum Comments to Analyze", 10, 200, 50)

    if analyze_button and video_url:
        video_id = extract_video_id(video_url)

        if not video_id:
            st.error("❌ Invalid YouTube URL. Please enter a valid YouTube video URL.")
        else:
            with st.spinner("🔄 Fetching video data..."):
                # Fetch video details
                video_data = fetch_video_details(video_id, api_key)

                if video_data:
                    st.session_state.video_data = video_data

                    # Fetch comments
                    comments = fetch_comments(video_id, api_key, max_comments)

                    if comments:
                        st.session_state.comments_data = comments
                        st.success(f"✅ Successfully fetched {len(comments)} comments!")
                    else:
                        st.warning("⚠️ No comments found or comments are disabled for this video.")

    # Display results if data is available
    if st.session_state.video_data:
        video_id = extract_video_id(video_url)
        video_data = st.session_state.video_data

        # Video Information Section
        st.header("📊 Video Information")

        col1, col2 = st.columns([1, 2])

        with col1:
            if video_data.get('thumbnail'):
                st.image(video_data['thumbnail'], width=300)

        with col2:
            st.markdown(f"**Title:** {video_data['title']}")
            st.markdown(f"**Channel:** {video_data['channel_name']}")
            st.markdown(f"**Published:** {video_data['published_date'][:10]}")

            # Metrics
            col2_1, col2_2, col2_3 = st.columns(3)
            with col2_1:
                st.metric("👀 Views", f"{video_data['view_count']:,}")
            with col2_2:
                st.metric("👍 Likes", f"{video_data['like_count']:,}")
            with col2_3:
                st.metric("💬 Comments", f"{video_data['comment_count']:,}")

        # Comments Analysis Section
        if st.session_state.comments_data:
            comments_data = st.session_state.comments_data

            st.header("💬 Comments Analysis")

            with st.spinner("🔄 Analyzing sentiments and toxicity..."):
                # Prepare comments dataframe
                df_comments = pd.DataFrame(comments_data)
                df_comments['cleaned_text'] = df_comments['text'].apply(clean_text)

                # Sentiment Analysis
                df_comments['sentiment_textblob'] = df_comments['cleaned_text'].apply(analyze_sentiment_textblob)
                df_comments['sentiment_vader'] = df_comments['cleaned_text'].apply(analyze_sentiment_vader)

                # Toxicity Detection
                if enable_toxicity:
                    with st.spinner("🔍 Detecting toxicity..."):
                        if toxicity_method == "Keyword-based Only":
                            df_comments['toxicity'] = detect_toxicity_simple(df_comments['cleaned_text'].tolist())
                        else:
                            df_comments['toxicity'] = detect_toxicity(df_comments['cleaned_text'].tolist())
                else:
                    df_comments['toxicity'] = 'Disabled'

            # Sentiment Analysis Results
            st.subheader("😊 Sentiment Analysis")

            # Choose sentiment method
            sentiment_method = st.selectbox("Choose Sentiment Analysis Method:",
                                            ["TextBlob", "VADER"])

            sentiment_col = 'sentiment_textblob' if sentiment_method == 'TextBlob' else 'sentiment_vader'
            sentiment_counts = df_comments[sentiment_col].value_counts().to_dict()

            # Display sentiment metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("😊 Positive", sentiment_counts.get('Positive', 0))
            with col2:
                st.metric("😐 Neutral", sentiment_counts.get('Neutral', 0))
            with col3:
                st.metric("😞 Negative", sentiment_counts.get('Negative', 0))

            # Sentiment Chart
            fig_sentiment = create_sentiment_chart(sentiment_counts)
            st.plotly_chart(fig_sentiment, use_container_width=True)

            # Toxicity Analysis
            if enable_toxicity and 'Disabled' not in df_comments['toxicity'].values:
                st.subheader("🚨 Toxicity Analysis")
                toxicity_counts = df_comments['toxicity'].value_counts().to_dict()

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🚨 Toxic Comments", toxicity_counts.get('Toxic', 0))
                with col2:
                    st.metric("✅ Non-toxic Comments", toxicity_counts.get('Non-toxic', 0))
                with col3:
                    st.metric("❓ Unknown", toxicity_counts.get('Unknown', 0))

                # Toxicity Chart
                if len([k for k in toxicity_counts.keys() if k != 'Unknown']) > 0:
                    fig_toxicity = px.bar(
                        x=list(toxicity_counts.keys()),
                        y=list(toxicity_counts.values()),
                        title="Toxicity Distribution",
                        color=list(toxicity_counts.keys()),
                        color_discrete_map={
                            'Toxic': '#FF4444',
                            'Non-toxic': '#44AA44',
                            'Unknown': '#CCCCCC'
                        }
                    )
                    st.plotly_chart(fig_toxicity, use_container_width=True)
            elif not enable_toxicity:
                st.info("💡 Toxicity detection is disabled. Enable it in the sidebar to see toxicity analysis.")

            # Word Cloud
            st.subheader("☁️ Word Cloud")
            wordcloud_fig = create_wordcloud(df_comments['cleaned_text'].tolist())
            if wordcloud_fig:
                st.pyplot(wordcloud_fig)

            # Sample Comments
            st.subheader("📝 Sample Comments")

            # Filter options
            filter_option = st.selectbox("Filter Comments:", ["All", "Positive", "Negative", "Neutral"])

            if filter_option != "All":
                filtered_df = df_comments[df_comments[sentiment_col] == filter_option]
            else:
                filtered_df = df_comments

            # Display comments
            for idx, row in filtered_df.head(10).iterrows():
                with st.expander(f"Comment by {row['author']} - {row[sentiment_col]}"):
                    st.write(f"**Text:** {row['text']}")
                    st.write(f"**Likes:** {row['like_count']}")
                    st.write(f"**Sentiment:** {row[sentiment_col]}")
                    if enable_toxicity and row['toxicity'] != 'Disabled':
                        st.write(f"**Toxicity:** {row['toxicity']}")
                    st.write(f"**Published:** {row['published_at'][:10]}")

            # Download Results
            st.subheader("⬇️ Download Results")
            csv = df_comments.to_csv(index=False)
            st.download_button(
                label="Download Comments Analysis as CSV",
                data=csv,
                file_name=f"youtube_comments_analysis_{video_id}.csv",
                mime="text/csv"
            )

# Footer
st.markdown("---")
st.markdown("Created with ❤️ using Streamlit | YouTube Data API v3 | Various NLP Libraries")