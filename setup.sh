mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"angel.martinez@celeris-systems.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml