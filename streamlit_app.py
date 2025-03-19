import streamlit as st
import pandas as pd
import re
import torch
import nltk
import os
import ssl
from typing import Dict, Tuple
from nltk.tokenize import PunktSentenceTokenizer
from transformers import BertTokenizer, BertModel, BartTokenizer, BartForConditionalGeneration, AutoTokenizer

# Download required NLTK data
#try:
#    nltk.data.find('tokenizers/PunktSentenceTokenizer')
#except LookupError:
#    nltk.download('PunktSentenceTokenizer')

# Model Loading with Caching
@st.cache_resource
def load_bart_models():
    """Load BART models with caching"""
    try:
        model_name = "facebook/bart-large-cnn"
        bart_tokenizer = BartTokenizer.from_pretrained(model_name)
        bart_model = BartForConditionalGeneration.from_pretrained(model_name)
        return bart_tokenizer, bart_model
    except Exception as e:
        st.error(f"Error loading BART models: {str(e)}")
        return None, None

@st.cache_resource
def load_finetuned_models():
    """Load fine-tuned models with caching"""
    try:
        def load_bert_from_pretrained(model_path):
            """Load BERT model from saved pretrained directory"""
            try:
                model = BertModel.from_pretrained(model_path)
                tokenizer = BertTokenizer.from_pretrained(model_path)
                return model, tokenizer
            except Exception as e:
                st.error(f"Error loading model from {model_path}: {str(e)}")
                return None, None
        
        # Load models from saved directories
        with st.spinner("Loading neutral sentiment model..."):
            net_model, net_tokenizer = load_bert_from_pretrained('finetuned_bert_neutral')
            
        with st.spinner("Loading negative sentiment model..."):
            neg_model, neg_tokenizer = load_bert_from_pretrained('finetuned_bert_negative')
            
        with st.spinner("Loading base BERT model for positive sentiment..."):
            pos_model, pos_tokenizer = load_bert_from_pretrained('finetuned_bert_positive')
        
        # Move models to appropriate device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if net_model:
            net_model = net_model.to(device)
        if neg_model:
            neg_model = neg_model.to(device)
        pos_model = pos_model.to(device)
        
        st.success("All models loaded successfully!")
        return pos_model, net_model, neg_model, pos_tokenizer, net_tokenizer, neg_tokenizer
        
    except Exception as e:
        st.error(f"Error loading fine-tuned models: {str(e)}")
        return None, None, None, None, None, None
        
# Load all models at startup
bart_tokenizer, bart_model = load_bart_models()
pos_model, net_model, neg_model, pos_tokenizer, net_tokenizer, neg_tokenizer = load_finetuned_models()

@st.cache_data
def clean_text(text: str) -> str:
    # Normalize the text (convert to lowercase)
    text = text.lower()

    # Replace typos with word boundaries to ensure replacement occurs only when surrounded by spaces
    text = re.sub(r'\badv\b', 'advertisement', text)
    text = re.sub(r'\bads\b', 'advertisement', text)
    text = re.sub(r'\bupreally\b', 'up really', text)
    text = re.sub(r'\bntap\b', 'great', text)
    text = re.sub(r'\bn\b', 'and', text)
    text = re.sub(r'\bgreat markotop\b', 'very good', text)
    text = re.sub(r'\btopmarkotop\b', 'very good', text)
    text = re.sub(r'\bunitthanksother\b', 'unit thanks other', text)
    text = re.sub(r'\bapt\b', 'apartment', text)
    text = re.sub(r'\baprt\b', 'apartment', text)
    text = re.sub(r'\bgc\b', 'fast', text)
    text = re.sub(r'\bsatset\b', 'fast', text)
    text = re.sub(r'\bapk\b', 'application', text)
    text = re.sub(r'\bapp\b', 'application', text)
    text = re.sub(r'\bapps\b', 'application', text)
    text = re.sub(r'\bgoib\b', 'hidden', text)
    text = re.sub(r'\bthx u\b', 'thankyou', text)
    text = re.sub(r'\bthx\b', 'thanks', text)
    text = re.sub(r'\brmboy\b', 'roomboy', text)
    text = re.sub(r'\bmantaaaap\b', 'excellent', text)
    text = re.sub(r'\btop\b', 'excellent', text)
    text = re.sub(r'\bops\b', 'operations', text)
    text = re.sub(r'\bpeni\b', '', text)
    text = re.sub(r'\bdisappointingthe\b', 'disappointing the', text)
    text = re.sub(r'\bcs\b', 'customer service', text)
    text = re.sub(r'\bbtw\b', 'by the way', text)
    text = re.sub(r'\b2023everything\b', '2023 everything', text)
    text = re.sub(r'\b2023its\b', '2023 its', text)
    text = re.sub(r'\bbadthe\b', 'bad the', text)
    text = re.sub(r'\bphotothe\b', 'photo the', text)
    text = re.sub(r'\bh-1\b', 'the day before', text)
    text = re.sub(r'\bac\b', 'air conditioner', text)
    text = re.sub(r'\b30-60\b', '30 to 60', text)
    text = re.sub(r'\b8-9\b', '8 to 9', text)
    text = re.sub(r'\bgb/day\b', 'gb per day', text)
    text = re.sub(r'\bnamethe\b', 'name the', text)
    text = re.sub(r'\bluv\b', 'love', text)
    text = re.sub(r'\bc/i\b', 'checkin', text)
    text = re.sub(r'\+', 'and', text)
    text = re.sub(r'\bwfh\b', 'work from home', text)
    text = re.sub(r'\btl\b', 'team leader', text)
    text = re.sub(r'\bspv\b', 'supervisor', text)
    text = re.sub(r'\b2.5hrs\b', '2 and a half hours', text)
    text = re.sub(r'\b&\b', 'and', text)

    # Remove special characters but keep numbers
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove repeating characters
    text = re.sub(r'(\b\w*?)(\w)\2{2,}(\w*\b)', r'\1\2\3', text)

    return text.strip()  # Ensure no leading/trailing whitespace

# Optimized Data Loading
@st.cache_data
def get_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and cache data from Excel files"""
    try:
        df_pos = pd.read_excel("cat_positive.xlsx")
        df_net = pd.read_excel("cat_neutral.xlsx")
        df_neg = pd.read_excel("cat_negative.xlsx")
        
        # Create copies with only required columns
        df_pos_copy = df_pos[['reviews']].copy()
        df_net_copy = df_net[['reviews']].copy()
        df_neg_copy = df_neg[['reviews']].copy()
        
        # Add prep_reviews column
        df_pos_copy['prep_reviews'] = df_pos_copy['reviews'].copy()
        df_net_copy['prep_reviews'] = df_net_copy['reviews'].copy()
        df_neg_copy['prep_reviews'] = df_neg_copy['reviews'].copy()
        
        return df_pos_copy, df_net_copy, df_neg_copy
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None

# Initialize session state for DataFrames if not exists
if 'df_pos' not in st.session_state:
    df_pos, df_net, df_neg = get_data()
    st.session_state.df_pos = df_pos
    st.session_state.df_net = df_net
    st.session_state.df_neg = df_neg

@torch.no_grad()
def extract_important_sentences(model, tokenizer, original_sentences, top_k):
    """Extract important sentences using the fine-tuned BERT model"""
    try:
        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Initialize list to store sentence scores
        sentence_scores = []

        # Process each sentence through the fine-tuned BERT
        for i, sentence in enumerate(original_sentences):
            try:
                # Tokenize the sentence
                inputs = tokenizer(sentence,
                                return_tensors="pt",
                                max_length=512,
                                truncation=True,
                                padding=True)

                # Move inputs to the same device as model
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Get embeddings from the model
                outputs = model(**inputs, output_hidden_states=True)
                # Get the last hidden state from the tuple of hidden states
                embeddings = outputs.hidden_states[-1][0]  # Get the token embeddings from last layer

                # Calculate sentence score using max pooling followed by mean
                max_values_per_sentence = embeddings.max(dim=0).values
                mean_value_per_sentence = torch.mean(max_values_per_sentence)

                # Store score and index
                sentence_scores.append((mean_value_per_sentence.item(), i))
            
            except Exception as e:
                st.warning(f"Error processing sentence {i}: {str(e)}")
                continue

        # Handle case where no valid scores were generated
        if not sentence_scores:
            return original_sentences[:top_k]

        # Sort sentences by score in descending order
        sentence_scores.sort(reverse=True, key=lambda x: x[0])

        # Get top k sentences
        top_indices = [index for _, index in sentence_scores[:top_k]]
        important_sentences = [original_sentences[index] for index in top_indices]

        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return important_sentences
        
    except Exception as e:
        st.error(f"Error in extract_important_sentences: {str(e)}")
        return original_sentences[:top_k]  # Fallback to first k sentences

# Optimized BART Summarization
def bart_summarize(
    text: str,
    tokenizer: BartTokenizer,
    model: BartForConditionalGeneration,
    max_length: int = 150,
    min_length: int = 50,
    length_penalty: float = 2.0,
    no_repeat_ngram_size: int = 2,
    num_beams: int = 8,
    nucleus_sampling: float = 1.00,
    top_k: int = 100,
    temperature: float = 0.98,
    repetition_penalty: float = 1.0,
    early_stopping: bool = True,
    do_sample: bool = True,
    truncation: bool = True
) -> str:
    """Generate summary using BART with chunking for long texts"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Split text into chunks if too long
    max_chunk_length = 1024
    chunks = [
        text[i:i + max_chunk_length] 
        for i in range(0, len(text), max_chunk_length)
    ]
    
    summaries = []
    for chunk in chunks:
        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            max_length=1024,
            truncation=truncation
        )
        inputs = inputs.to(device)
        
        with torch.no_grad():
            summary_ids = model.generate(
                inputs["input_ids"],
                max_length=max_length,
                min_length=min_length,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                num_beams=num_beams,
                top_k=top_k,
                top_p=nucleus_sampling,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                early_stopping=early_stopping,
                do_sample=do_sample
            )
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summaries.append(summary)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return ' '.join(summaries)

# Session state initialization
if 'sidebar_state' not in st.session_state:
    st.session_state.sidebar_state = 'expanded'

# Sidebar
with st.sidebar:
    menu = st.radio(
        "",
        ["👋 Hello", "📊 View Data", "📝 View Summarization", "✨ Make Summarization"],
        label_visibility="collapsed"
    )

# Main content
if menu == "👋 Welcome":
    st.title("Travelio Reviews Summarizer")
    st.write(
        """
        👋 Welcome to the **Travelio Reviews Summarizer**!  
        This tool leverages advanced natural language processing models (BERT and BART) to provide concise summaries of Travelio reviews.  
        Whether you're analyzing customer feedback or seeking quick insights, this summarizer has got you covered.
        """
    )
    st.write("To get started, select a feature from the menu on the left.")


elif menu == "✨ Make Summarization":
    st.title("Automatic Travelio Reviews Summarizer")
    
    category = st.selectbox('Select review category:', ['Positive', 'Neutral', 'Negative'])
    text = st.text_input('Enter your review here:')
    
    if st.button('Summarize'):
        # Check if models are loaded
        if None in [pos_model, net_model, neg_model, pos_tokenizer, net_tokenizer, neg_tokenizer]:
            st.error("Models are not properly loaded. Please check the model files.")

        with st.spinner("Processing review..."):
            # Clean the new text
            cleaned_text = clean_text(text)
            
            if not text:
                st.warning("Please enter some text to summarize.")
            
            # Add new review to appropriate DataFrame
            new_row = pd.DataFrame({'reviews': [text], 'prep_reviews': [cleaned_text]})
            
            if category == 'Positive':
                st.session_state.df_pos = pd.concat([st.session_state.df_pos, new_row], ignore_index=True)
                current_df = st.session_state.df_pos
                model_to_use = pos_model
                tokenizer_to_use = pos_tokenizer
            elif category == 'Neutral':
                st.session_state.df_net = pd.concat([st.session_state.df_net, new_row], ignore_index=True)
                current_df = st.session_state.df_net
                model_to_use = net_model
                tokenizer_to_use = net_tokenizer
            else:
                st.session_state.df_neg = pd.concat([st.session_state.df_neg, new_row], ignore_index=True)
                current_df = st.session_state.df_neg
                model_to_use = neg_model
                tokenizer_to_use = neg_tokenizer

            if model_to_use is None:
                st.error(f"Model for {category} category is not available")

            # Process all sentences in the DataFrame
            all_sentences = []
            for review in current_df['prep_reviews']:
                sentences = tokenize(str(review))
                all_sentences.extend(sentences)
                
            with st.spinner("Analyzing important sentences..."):
                important_sentences = extract_important_sentences(
                    model_to_use,
                    tokenizer_to_use,
                    all_sentences,
                    top_k=5
                )
                
            combined_text = " ".join(important_sentences)
            
            with st.spinner("Generating summary..."):
                if category == 'Neutral':
                    summary = bart_summarize(
                        combined_text,
                        bart_tokenizer,
                        bart_model,
                        max_length=50,
                        min_length=20
                    )
                else:
                    summary = bart_summarize(
                        combined_text,
                        bart_tokenizer,
                        bart_model,
                        max_length=150,
                        min_length=50
                )
            
            st.success("Summary generated successfully!")
            st.write(summary)

elif menu == "📝 View Summarization":
    st.title("Pre-generated Summaries")
    category = st.selectbox('Select summary category:', ['Positive', 'Neutral', 'Negative'])
    
    if st.button('Show Summary'):
        summaries = {
            'Positive': """This application is very helpful i don't know if someone said it wasn't enough but its fine for me travelio admin is also very communicative after we make an order. The service from novandri is very good the room is clean and appropriate. It's very suitable for vacation or business.""",
            'Neutral': """It's really easy to find monthly apartment rentals the unit is also complete clean just live in it practical for those who suddenly need it. It's reliable and the prices are quite competitive recommended it's just that if there is a chat notification tap it when you enter the application you have to restart from the beginning.""",
            'Negative': """Customer service really slow response making it difficult to communicate in real time not informed that parking access cannot be arranged on holidays it can only be done on mondays whereas paying per hour is expensive 24 hours x 50 alone. Tenants are made unable to extend or cannot rent the same unit again unless they pay an additional fine of 1 million."""
        }
        st.write(f"**Summary for {category} reviews:**")
        st.write(summaries[category])

elif menu == "📊 View Data":
    st.title('Travelio Reviews Data')
    
    category = st.selectbox('Select data category:', ['Positive', 'Neutral', 'Negative'])
    
    @st.cache_data
    def get_displayed_data(df: pd.DataFrame, n_rows: int = 1000) -> pd.DataFrame:
        """Get subset of data for display"""
        return df.head(n_rows)
    
    if category == 'Positive' and st.session_state.df_pos is not None:
        st.write(f"Total reviews: {len(st.session_state.df_pos)}")
        st.dataframe(
            get_displayed_data(st.session_state.df_pos['reviews']),
            height=400,
            use_container_width=True
        )
    elif category == 'Neutral' and st.session_state.df_net is not None:
        st.write(f"Total reviews: {len(st.session_state.df_net)}")
        st.dataframe(
            get_displayed_data(st.session_state.df_net['reviews']),
            height=400,
            use_container_width=True
        )
    elif category == 'Negative' and st.session_state.df_neg is not None:
        st.write(f"Total reviews: {len(st.session_state.df_neg)}")
        st.dataframe(
            get_displayed_data(st.session_state.df_neg['reviews']),
            height=400,
            use_container_width=True
        )
