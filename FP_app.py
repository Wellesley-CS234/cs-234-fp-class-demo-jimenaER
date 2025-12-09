import streamlit as st
import pandas as pd
import plotly.express as px

# 1. Configuration and Data Loading
st.set_page_config(layout="wide", page_title="Spanish wikipedia articles data analyisi")

#Read dataframe Data

def load_data():
    """Loads the Titanic dataset and performs basic preprocessing for analysis."""
    # Load the well-known Titanic dataset from a public repository
    data_url = 'data/Sample_top20_country_articles.csv'
    df = pd.read_csv(data_url)

    # Basic feature engineering for easier plotting
    top_articles=df.reset_index(drop=True)
    most_frequent= top_articles.groupby('country')['assigned label'].agg(lambda x : x.mode()[0])
    category_map_data= most_frequent.reset_index()
    category_map_data.columns=['country','most_frequent_category']

    return category_map_data

df = load_data()

# 2. Sidebar Menu Setup
st.sidebar.title("Select Analysis")
analysis_options = {
    "1": "Top article categories (map Chart)"}

option_key = st.sidebar.radio(
    "Choose an exploratory view:",
    list(analysis_options.keys()),
    format_func=lambda x: analysis_options[x]
)

st.title(f"Spanish wikipedia articles: {analysis_options[option_key]}")

# 3. Main Page Logic based on Selection
if option_key == "1":
    # --- Analysis 1: Top article categories of top 20 article per country (map Chart) ---
    st.markdown('**RQ: What categories of articles do spanish speaking countries interact with the most?**')

    st.header("Categories of top 20 viewed Wikipedia articles per country in 2023-02")
    st.markdown("Examine the top categories of spanish articles read across different countries.")
    st.write('Data was categorized using the descriptions components and each article and then performing zero shot classification for broader categories. ')
    st.write('It should be noted that this is sample data for only a month, due to this some contries are not highlighted in the data.')
    st.write('Additionaly, through ' \
    'this phase we also excluded the data for missing wikipedia descriptions, which will be categorized and displayed in the next figures of this project, and compared with the error of the zero shot classifiction')

    fig1= px.choropleth(
    df,
    locations='country',
    locationmode='country names',
    color= 'most_frequent_category',
    scope='world',
    title='Top article categories of top 20 article per country',
    height=800)

    st.plotly_chart(fig1, use_container_width=True)
    st.caption("Figure 1: This map illustrates the most dominant category of articles in the top 20 viewed articles for each country based on the 2023-02 sample data." \
    "Results of this sample data reject our hypothesis")
    
    st.markdown("Snippet of data used to represent categories")
    
    full= pd.read_csv('data/Sample_top20_country_articles.csv')
    st.dataframe(full.head(10))