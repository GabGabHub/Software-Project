import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
import seaborn as sns


def home_page():
    st.title("Home Page")
    df = pd.read_csv('quality_of_life_indices_by_country.csv')
    st.session_state.df = df  # Store the DataFrame
    st.dataframe(df)

def page1():
    st.title("Page 1")
    st.title("CLIENTS_LEASING500 Statistical Analysis")  # Title for the page

    # Access the DataFrame from session state
    if "df" in st.session_state:
        dfm = st.session_state.df

        # Display descriptive statistics
        st.header("1. Pandas Methods")
        st.write(dfm.describe())

        # Display histograms
        st.subheader("Histograms")
        dfm.hist()
        st.pyplot(plt)

        # Display kurtosis
        st.subheader("Kurtosis")
        numeric_dfm = dfm.select_dtypes(include=np.number)
        st.write(numeric_dfm.kurtosis())

        # Display skewness
        st.subheader("Skewness")
        st.write(numeric_dfm.skew())

        # Display analysis using scipy
        st.header("2. Scipy Package")
        y = dfm['REQUESTED_AMOUNT']
        x = dfm['AGE']

        # Function to display statistics and histogram
        def display_stats_hist(data, title):
            fig, ax = plt.subplots()
            ax.hist(data, bins='auto')
            ax.set_title(title)
            st.pyplot(fig)
            st.write(f"-------------Statistics for {title}-------------")
            st.write("average: ", np.mean(data))
            st.write("variance  : ", np.var(data))
            st.write("skewness : ", skew(data))
            st.write("kurtosis : ", kurtosis(data))

        # Display for REQUESTED_AMOUNT
        display_stats_hist(y, 'REQUESTED_AMOUNT')

        # Display for AGE
        display_stats_hist(x, 'AGE')

    else:
        st.write("Please upload a file on the home page.")


def page2():
    st.title("Page 2")
    st.title("Correlation Analysis")

    # Access the DataFrame from session state
    if "df" in st.session_state:
        dfm = st.session_state.df

        # Select numeric columns
        numeric_dfm = dfm.select_dtypes(include=['number'])

        # Calculate correlation
        dfm_corr = numeric_dfm.corr()

        # Display correlation matrix
        st.write("## Correlation Matrix")
        st.write(dfm_corr)

        # Scatter plot options
        st.write("## Scatter Plot")

        # Create and display scatter plot
        fig, ax = plt.subplots()
        ax.scatter(dfm["AGE"], dfm["REQUESTED_AMOUNT"])
        ax.set_xlabel("AGE")
        ax.set_ylabel("REQUESTED_AMOUNT")
        st.pyplot(fig)

        fig, ax = plt.subplots()
        ax.scatter(dfm["FIDELITY"], dfm["DEPOSIT_AMOUNT"])
        ax.set_xlabel("FIDELITY")
        ax.set_ylabel("DEPOSIT_AMOUNT")
        st.pyplot(fig)

        # Pairplot
        st.write("## Pairplot")

        pairplot_fig = sns.pairplot(dfm[dfm['REQUESTED_AMOUNT'] > 100000],
                                     vars=['AGE', 'VAL_CREDITS_RON', 'REQUESTED_AMOUNT', 'INCOME_PER_YEAR_RON'],
                                     hue='PRESCORING', diag_kind='hist')

        st.pyplot(plt)
        # Heatmap
        st.write("## Heatmap")

        fig, ax = plt.subplots(figsize=(10, 6))  # Create new figure
        sns.heatmap(numeric_dfm.corr(), annot=True, ax=ax)  # Assign to ax
        st.pyplot(fig)  # Use newly created figure
    else:
        st.write("Please upload a file on the home page.")


if "page" not in st.session_state:
    st.session_state.page = "Home"  # Set initial page to "Home"

st.markdown(
    """
    <style>
    div.stButton > button:first-child {
        background-color: #8FD3FE;
        color: #CC8899;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar buttons
if st.sidebar.button("Home"):
    st.session_state.page = "Home"
if st.sidebar.button("Histograms"):
    st.session_state.page = "Histograms"
if st.sidebar.button("Scatter Plots"):
    st.session_state.page = "Scatter Plots"

# Display content based on session state
if st.session_state.page == "Home":
    home_page()
elif st.session_state.page == "Histograms":
    page1()
elif st.session_state.page == "Scatter Plots":
    page2()