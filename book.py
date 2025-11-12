# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import time

# Custom CSS for beautiful styling
def load_css():
    st.markdown("""
    <style>
    /* Responsive Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        min-width: 220px;
        max-width: 300px;
        border-radius: 0 20px 20px 0;
        box-shadow: 2px 0 20px rgba(102,126,234,0.08);
        padding-top: 1.5rem;
        transition: all 0.3s ease;
    }
    [data-testid="stSidebar"] .css-1d391kg, [data-testid="stSidebar"] .css-1v0mbdj {
        color: white !important;
    }
    @media (max-width: 900px) {
        [data-testid="stSidebar"] {
            min-width: 120px;
            max-width: 180px;
            font-size: 0.9rem;
            padding-top: 0.5rem;
        }
    }
    @media (max-width: 600px) {
        [data-testid="stSidebar"] {
            min-width: 60px;
            max-width: 100px;
            font-size: 0.8rem;
            padding-top: 0.2rem;
        }
    }
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {
        padding-top: 0rem;
    }
    
    /* Custom Header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        animation: fadeInDown 1s ease-out;
    }
    
    .main-header h1 {
        color: white;
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        font-size: 3rem;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    /* Custom Cards */
    .book-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
        transition: all 0.3s ease;
        animation: slideInUp 0.6s ease-out;
    }
    
    .book-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.15);
    }
    
    .book-title {
        font-family: 'Poppins', sans-serif;
        font-size: 1.3rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    
    .book-author {
        color: #7f8c8d;
        font-style: italic;
        margin-bottom: 0.5rem;
    }
    
    .book-category {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        color: #333;
        padding: 0.2rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 500;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    
    .book-description {
        color: #666;
        font-size: 0.9rem;
        line-height: 1.4;
        margin-bottom: 0.8rem;
        font-style: italic;
    }
    
    .book-rating {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-weight: 500;
        display: inline-block;
        font-size: 0.9rem;
    }
    
    .similarity-score {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    
    .analysis-box {
        background: linear-gradient(135deg, #e0c3fc 0%, #9bb5ff 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: #333;
    }
    
    /* Custom Input Styling */
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e1e8ed;
        padding: 0.75rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25);
    }
    
    /* Custom Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        font-family: 'Poppins', sans-serif;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Loading Animation */
    .loading-container {
        text-align: center;
        padding: 2rem;
    }
    
    .loading-spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #667eea;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 0 auto;
    }
    
    /* Animations */
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Success/Error Messages */
    .success-message {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 500;
        animation: fadeInDown 0.5s ease-out;
    }
    
    .error-message {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 500;
        animation: fadeInDown 0.5s ease-out;
    }
    
    /* Hide Streamlit Style */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        .book-card {
            padding: 1rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# Load dataset
@st.cache_data
def load_books():
    try:
        # Load from local CSV file
        books = pd.read_csv('books.csv')
        # Keep required columns (removed thumbnail)
        required_columns = ['title', 'authors', 'average_rating', 'categories', 'description', 
                           'published_year', 'num_pages', 'ratings_count']
        # Check if all required columns exist
        missing_columns = [col for col in required_columns if col not in books.columns]
        if missing_columns:
            st.error(f"Missing columns in books.csv: {missing_columns}")
            return pd.DataFrame()
        books = books[required_columns].copy()
        # Clean the data
        books = books.dropna(subset=['title', 'authors', 'average_rating'])
        # Convert columns to appropriate types
        books['average_rating'] = pd.to_numeric(books['average_rating'], errors='coerce')
        books['published_year'] = pd.to_numeric(books['published_year'], errors='coerce')
        books['num_pages'] = pd.to_numeric(books['num_pages'], errors='coerce')
        books['ratings_count'] = pd.to_numeric(books['ratings_count'], errors='coerce')
        # Fill missing descriptions with title + categories
        books['description'] = books['description'].fillna(books['title'] + ' ' + books['categories'].fillna(''))
        # Remove rows where essential data is missing
        books = books.dropna(subset=['average_rating'])
        # Create rich semantic text combining multiple fields
        books['semantic_text'] = (
            books['title'].astype(str) + ' by ' + 
            books['authors'].astype(str) + '. ' +
            'Category: ' + books['categories'].fillna('General').astype(str) + '. ' +
            'Description: ' + books['description'].astype(str)
        )
        return books
    except Exception as e:
        st.error(f"Error loading books.csv: {str(e)}")
        # Return empty dataframe as fallback
        return pd.DataFrame(columns=['title', 'authors', 'average_rating', 'categories', 
                                   'description', 'published_year', 'num_pages', 'ratings_count', 
                                   'semantic_text'])

# Semantic-based recommendation system
@st.cache_resource
def load_semantic_model():
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except Exception as e:
        st.error(f"Could not load semantic model: {str(e)}")
        return None

@st.cache_data
def precompute_embeddings(_books):
    """Precompute book embeddings for faster recommendations"""
    model = load_semantic_model()
    try:
        if model is None or len(_books) == 0:
            return None
        # Create rich semantic representations
        semantic_texts = _books['semantic_text'].tolist()
        with st.spinner("Computing book embeddings... This may take a moment."):
            embeddings = model.encode(semantic_texts, convert_to_tensor=False, show_progress_bar=False)
        return embeddings
    except Exception as e:
        st.error(f"Error computing embeddings: {str(e)}")
        return None

def analyze_emotional_tone(text):
    """Analyze emotional tone of user query"""
    emotional_keywords = {
        'happy': ['happy', 'joy', 'cheerful', 'uplifting', 'positive', 'funny', 'humor', 'comedy'],
        'sad': ['sad', 'melancholy', 'tragic', 'heartbreaking', 'emotional', 'tears', 'grief'],
        'exciting': ['adventure', 'action', 'thrilling', 'exciting', 'fast-paced', 'suspense'],
        'romantic': ['love', 'romance', 'romantic', 'relationship', 'dating', 'marriage'],
        'mysterious': ['mystery', 'detective', 'crime', 'investigation', 'puzzle', 'secret'],
        'dark': ['dark', 'horror', 'scary', 'gothic', 'psychological', 'disturbing'],
        'inspiring': ['inspiring', 'motivational', 'uplifting', 'hopeful', 'empowering'],
        'peaceful': ['calm', 'peaceful', 'serene', 'quiet', 'meditation', 'spiritual']
    }
    text_lower = text.lower()
    detected_tones = []
    for tone, keywords in emotional_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            detected_tones.append(tone)
    return detected_tones if detected_tones else ['neutral']

def extract_categories(text):
    """Extract likely book categories from user query"""
    category_keywords = {
        'Fiction': ['fiction', 'novel', 'story', 'narrative'],
        'Romance': ['romance', 'love', 'romantic', 'relationship'],
        'Mystery': ['mystery', 'detective', 'crime', 'thriller'],
        'Science Fiction': ['sci-fi', 'science fiction', 'space', 'future', 'technology'],
        'Fantasy': ['fantasy', 'magic', 'wizard', 'dragon', 'medieval'],
        'Horror': ['horror', 'scary', 'ghost', 'supernatural'],
        'Biography': ['biography', 'memoir', 'autobiography', 'life story'],
        'History': ['history', 'historical', 'past', 'ancient'],
        'Self Help': ['self help', 'improvement', 'motivation', 'guide'],
        'Business': ['business', 'entrepreneurship', 'leadership', 'management']
    }
    text_lower = text.lower()
    detected_categories = []
    for category, keywords in category_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            detected_categories.append(category)
    return detected_categories if detected_categories else ['General']

def semantic_recommend(user_input, books, book_embeddings, top_k=5, category_filter=None, emotional_tone=None):
    """Advanced semantic recommendation with category and emotional tone"""
    try:
        model = load_semantic_model()
        if model is None or book_embeddings is None:
            return [], [], {}, "Semantic model not available."
        # Analyze user input
        detected_tones = analyze_emotional_tone(user_input)
        detected_categories = extract_categories(user_input)
        # Enhance user query with emotional and category context
        enhanced_query = user_input
        if emotional_tone:
            enhanced_query += f" {emotional_tone} mood"
        if category_filter:
            enhanced_query += f" {category_filter} category"
        # Add detected context
        enhanced_query += f" {' '.join(detected_tones)} tone {' '.join(detected_categories)} genre"
        # Get query embedding
        query_embedding = model.encode(enhanced_query, convert_to_tensor=False)
        # Calculate similarities
        similarities = cosine_similarity([query_embedding], book_embeddings)[0]
        # Apply filters if specified
        filtered_indices = list(range(len(books)))
        if category_filter and category_filter != "Any":
            filtered_indices = [i for i in filtered_indices 
                              if category_filter.lower() in str(books.iloc[i]['categories']).lower()]
        # Get top recommendations from filtered results
        if filtered_indices:
            filtered_similarities = [(i, similarities[i]) for i in filtered_indices]
            filtered_similarities.sort(key=lambda x: x[1], reverse=True)
            top_indices = [idx for idx, _ in filtered_similarities[:top_k] if idx < len(books)]
        else:
            import numpy as np
            sorted_indices = np.argsort(similarities)[::-1]
            top_indices = [idx for idx in sorted_indices[:top_k] if idx < len(books)]
        recommended_books = books.iloc[top_indices].copy() if top_indices else books.iloc[[]].copy()
        scores = [similarities[i] for i in top_indices]
        # Add analysis results
        analysis = {
            'detected_tones': detected_tones,
            'detected_categories': detected_categories,
            'enhanced_query': enhanced_query
        }
        return recommended_books, scores, analysis, "success"
    except Exception as e:
        return [], [], {}, f"Error in semantic recommendation: {str(e)}"

# Custom UI Components
def create_beautiful_header():
    st.markdown("""
    <div class="main-header">
        <h1>📚 Book Recommendation System</h1>
        <p>Discover your next favorite book with AI-powered recommendations</p>
    </div>
    """, unsafe_allow_html=True)

def create_book_card(book, index, similarity_score=None, analysis=None):
    """Create a beautiful book card without images"""
    rating_stars = "⭐" * int(book['average_rating'])
    # Extract category for display
    category = book.get('categories', 'General')
    if pd.isna(category):
        category = 'General'
    else:
        # Truncate long categories
        category = category[:25] + "..." if len(str(category)) > 25 else str(category)
    # Get description snippet
    description = book.get('description', '')
    if pd.isna(description):
        description = book['title']
    description_snippet = str(description)[:150] + "..." if len(str(description)) > 150 else str(description)
    # Similarity score display
    similarity_display = ""
    if similarity_score is not None:
        similarity_display = f'<div class="similarity-score">🎯 Match: {similarity_score:.1%}</div>'
    
    card_html = f"""
    <div class="book-card" style="animation-delay: {index * 0.1}s;">
        <div class="book-title">📖 {book['title']}</div>
        <div class="book-author">✍️ by {book['authors']}</div>
        <div class="book-category">📂 {category}</div>
        <div class="book-description">{description_snippet}</div>
        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 1rem; flex-wrap: wrap; gap: 0.5rem;">
            <div class="book-rating">{rating_stars} {book['average_rating']:.1f}/5.0</div>
            {similarity_display}
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

def show_loading_animation():
    """Show beautiful loading animation"""
    loading_html = """
    <div class="loading-container">
        <div class="loading-spinner"></div>
        <p style="margin-top: 1rem; color: #667eea; font-weight: 500;">
            🔍 Finding perfect books for you...
        </p>
    </div>
    """
    return st.markdown(loading_html, unsafe_allow_html=True)

def show_success_message(message):
    """Show success message with animation"""
    try:
        # Load from local CSV file
        books = pd.read_csv('books.csv')
        # Keep required columns (removed thumbnail)
        required_columns = ['title', 'authors', 'average_rating', 'categories', 'description', 
                           'published_year', 'num_pages', 'ratings_count']
        # Check if all required columns exist
        missing_columns = [col for col in required_columns if col not in books.columns]
        if missing_columns:
            st.error(f"Missing columns in books.csv: {missing_columns}")
            return pd.DataFrame()
        books = books[required_columns].copy()
        # Clean the data
        books = books.dropna(subset=['title', 'authors', 'average_rating'])
        # Convert columns to appropriate types
        books['average_rating'] = pd.to_numeric(books['average_rating'], errors='coerce')
        books['published_year'] = pd.to_numeric(books['published_year'], errors='coerce')
        books['num_pages'] = pd.to_numeric(books['num_pages'], errors='coerce')
        books['ratings_count'] = pd.to_numeric(books['ratings_count'], errors='coerce')
        # Fill missing descriptions with title + categories
        books['description'] = books['description'].fillna(books['title'] + ' ' + books['categories'].fillna(''))
        # Remove rows where essential data is missing
        books = books.dropna(subset=['average_rating'])
        # Create rich semantic text combining multiple fields
        books['semantic_text'] = (
            books['title'].astype(str) + ' by ' + 
            books['authors'].astype(str) + '. ' +
            'Category: ' + books['categories'].fillna('General').astype(str) + '. ' +
            'Description: ' + books['description'].astype(str)
        )
        return books
    except Exception as e:
        st.error(f"Error loading books.csv: {str(e)}")
    # Return empty dataframe as fallback
    return pd.DataFrame(columns=['title', 'authors', 'average_rating', 'categories', 
                   'description', 'published_year', 'num_pages', 'ratings_count', 
                   'semantic_text'])

# Main App
def main():
    # Streamlit App Configuration
    st.set_page_config(
        page_title="📚 Book Recommendation System",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    # Sidebar open instruction
    # Load custom CSS
    load_css()
    # Load books
    books = load_books()
    if len(books) == 0:
        st.error("No books could be loaded. Please check your books.csv file.")
        return
    # Beautiful Header
    create_beautiful_header()
    # Metrics Dashboard
    st.markdown("<br>", unsafe_allow_html=True)
    # --- Metrics Section (copied from previous working code) ---
    if len(books) == 0:
        st.error("No book data available. Please check your books.csv file.")
        return
    st.markdown("### 📊 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.5rem; border-radius: 15px; text-align: center; color: white;
                    box-shadow: 0 5px 20px rgba(0,0,0,0.1);">
            <h2 style="margin: 0; font-size: 2rem;">📚</h2>
            <h3 style="margin: 0.5rem 0; font-size: 1.5rem;">{len(books):,}</h3>
            <p style="margin: 0; opacity: 0.9;">Total Books</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        avg_rating = books['average_rating'].mean() if len(books) > 0 else 0
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 1.5rem; border-radius: 15px; text-align: center; color: white;
                    box-shadow: 0 5px 20px rgba(0,0,0,0.1);">
            <h2 style="margin: 0; font-size: 2rem;">⭐</h2>
            <h3 style="margin: 0.5rem 0; font-size: 1.5rem;">{avg_rating:.2f}</h3>
            <p style="margin: 0; opacity: 0.9;">Avg Rating</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        unique_authors = books['authors'].nunique() if len(books) > 0 else 0
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%); 
                    padding: 1.5rem; border-radius: 15px; text-align: center; color: white;
                    box-shadow: 0 5px 20px rgba(0,0,0,0.1);">
            <h2 style="margin: 0; font-size: 2rem;">✍️</h2>
            <h3 style="margin: 0.5rem 0; font-size: 1.5rem;">{unique_authors:,}</h3>
            <p style="margin: 0; opacity: 0.9;">Unique Authors</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        unique_categories = books['categories'].dropna().nunique() if 'categories' in books.columns else 0
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%); 
                    padding: 1.5rem; border-radius: 15px; text-align: center; color: white;
                    box-shadow: 0 5px 20px rgba(0,0,0,0.1);">
            <h2 style="margin: 0; font-size: 2rem;">📂</h2>
            <h3 style="margin: 0.5rem 0; font-size: 1.5rem;">{unique_categories:,}</h3>
            <p style="margin: 0; opacity: 0.9;">Categories</p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)

    # --- Top Navigation ---
    import matplotlib.pyplot as plt
    import seaborn as sns
    page = st.radio('Navigation', ['Book Recommender', 'Analytics'], horizontal=True)

    if page == 'Analytics':
        st.markdown("## 📈 Book Dataset Analytics")
        # Distribution of average ratings
        fig1, ax1 = plt.subplots()
        sns.histplot(books['average_rating'], bins=20, kde=True, ax=ax1, color='skyblue')
        ax1.set_title('Distribution of Average Ratings')
        ax1.set_xlabel('Average Rating')
        ax1.set_ylabel('Number of Books')
        st.pyplot(fig1)

        # Top 10 categories by book count
        if 'categories' in books.columns:
            top_categories = books['categories'].value_counts().head(10)
            fig2, ax2 = plt.subplots()
            sns.barplot(x=top_categories.values, y=top_categories.index, ax=ax2, palette='viridis')
            ax2.set_title('Top 10 Book Categories')
            ax2.set_xlabel('Number of Books')
            ax2.set_ylabel('Category')
            st.pyplot(fig2)

        # Top 10 authors by book count
        if 'authors' in books.columns:
            top_authors = books['authors'].value_counts().head(10)
            fig3, ax3 = plt.subplots()
            sns.barplot(x=top_authors.values, y=top_authors.index, ax=ax3, palette='mako')
            ax3.set_title('Top 10 Authors by Book Count')
            ax3.set_xlabel('Number of Books')
            ax3.set_ylabel('Author')
            st.pyplot(fig3)

        # Distribution of published years
        if 'published_year' in books.columns:
            fig4, ax4 = plt.subplots()
            sns.histplot(books['published_year'].dropna(), bins=30, kde=False, ax=ax4, color='salmon')
            ax4.set_title('Distribution of Published Years')
            ax4.set_xlabel('Published Year')
            ax4.set_ylabel('Number of Books')
            st.pyplot(fig4)
        return

    # Main Content
    st.markdown("""
    <div style="background: white; padding: 0.5rem; border-radius: 15px; 
               box-shadow: 0 5px 20px rgba(0,0,0,0.1); margin-bottom: 2rem;">
    """, unsafe_allow_html=True)
    # Enhanced Input Section
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.markdown("""
        <h3 style="color: #2c3e50; font-family: 'Poppins', sans-serif; 
                   margin-bottom: 1rem;">🔍 What are you looking for?</h3>
        """, unsafe_allow_html=True)
        
        user_input = st.text_input(
            "Search for books",
            value="",
            placeholder="Enter: 'Title' or 'Author Names', 'E.x: Dark mystery'....",
            help="Describe the book you want - include mood, genre, or specific themes",
            label_visibility="collapsed"
        )
    with col2:
        st.markdown("""
        <h3 style="color: #2c3e50; font-family: 'Poppins', sans-serif; 
                   margin-bottom: 1rem;">📂 Category</h3>
        """, unsafe_allow_html=True)
        # Get unique categories from the dataset
        available_categories = ['Any'] + sorted(books['categories'].dropna().unique().tolist()[:20])
        
        category_filter = st.selectbox(
            "Category Filter",
            available_categories,
            help="Filter by book category",
            label_visibility="collapsed"
        )
    with col3:
        st.markdown("""
        <h3 style="color: #2c3e50; font-family: 'Poppins', sans-serif; 
                   margin-bottom: 1rem;">😊 Mood</h3>
        """, unsafe_allow_html=True)
        
        emotional_tone = st.selectbox(
            "Emotional Tone",
            ["Any", "Happy", "Sad", "Exciting", "Romantic", "Mysterious", "Dark", "Inspiring", "Peaceful"],
            help="Choose the emotional tone you're looking for",
            label_visibility="collapsed"
        )
    # Number of recommendations slider
    num_recommendations = st.slider(
        "📊 Number of Recommendations",
        min_value=3,
        max_value=10,
        value=5,
        help="How many book recommendations would you like?"
    )
    # Precompute embeddings
    book_embeddings = precompute_embeddings(books)
    # Enhanced Button Section
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        if st.button("✨ Get AI Recommendations", use_container_width=True):
            if user_input.strip():
                # Show loading animation
                loading_placeholder = st.empty()
                with loading_placeholder.container():
                    show_loading_animation()
                # Simulate processing time for better UX
                time.sleep(1.5)
                loading_placeholder.empty()
                # Process recommendations using advanced semantic system
                try:
                    # Use the enhanced semantic recommendation
                    results, scores, analysis, status = semantic_recommend(
                        user_input, 
                        books,
                        book_embeddings,
                        top_k=num_recommendations,
                        category_filter=category_filter if category_filter != "Any" else None,
                        emotional_tone=emotional_tone.lower() if emotional_tone != "Any" else None
                    )
                    if status == "success" and len(results) > 0:
                        show_success_message(f"Found {len(results)} amazing recommendations for you!")
                        # Display analysis insights
                        if analysis:
                            st.markdown("""
                            <div class="analysis-box">
                                <h4 style="margin: 0 0 0.5rem 0;">🧠 AI Analysis of Your Request</h4>
                            """, unsafe_allow_html=True)
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.markdown(f"**🎭 Detected Mood:** {', '.join(analysis['detected_tones']).title()}")
                            with col_b:
                                st.markdown(f"**📚 Detected Genres:** {', '.join(analysis['detected_categories'])}")
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                        st.markdown("""
                        <h2 style="color: #2c3e50; font-family: 'Poppins', sans-serif; 
                                   text-align: center; margin: 2rem 0;">
                            📚 Your AI-Powered Recommendations
                        </h2>
                        """, unsafe_allow_html=True)
                        # Display books in beautiful enhanced cards
                        for index, (_, book) in enumerate(results.iterrows()):
                            similarity_score = scores[index] if index < len(scores) else None
                            create_book_card(book, index, similarity_score, analysis)
                        # Add enhanced stats about recommendations
                        avg_rec_rating = results['average_rating'].mean()
                        avg_similarity = sum(scores) / len(scores) if scores else 0
                        col_stat1, col_stat2 = st.columns(2)
                        with col_stat1:
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                       padding: 1rem; border-radius: 10px; text-align: center; 
                                       color: white; margin-top: 2rem;">
                                <p style="margin: 0; font-weight: 500;">
                                    ⭐ Average Rating: {avg_rec_rating:.2f}/5.0
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        with col_stat2:
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                                       padding: 1rem; border-radius: 10px; text-align: center; 
                                       color: white; margin-top: 2rem;">
                                <p style="margin: 0; font-weight: 500;">
                                    🎯 Average Match: {avg_similarity:.1%}
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.error("No matching books found. Try different keywords or remove filters!")
                except Exception as e:
                    st.error(f"Something went wrong: {str(e)}")
            else:
                st.error("Please describe what kind of book you're looking for!")
    st.markdown("</div>", unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 1rem; color: #7f8c8d; 
               font-style: italic; border-top: 1px solid #ecf0f1; margin-top: 5rem;">
        <p>✨ Made with ❤️ using Streamlit and AI • Happy Reading! 📖</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
