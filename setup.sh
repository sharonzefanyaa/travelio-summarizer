mkdir -p ~/.streamlit/

echo "\
[general]\n\
" > ~/.streamlit/config.toml

python -m nltk.downloader punkt
