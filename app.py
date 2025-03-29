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
    st.title("Quality of Life Statistical Analysis")  # Title for the page

    # Access the DataFrame from session state
    analysis_type = st.radio("Choose Analysis Type:", ("Pandas Methods", "Scipy Package"))
    dfm = st.session_state.df

    if analysis_type == "Pandas Methods":
        # Display descriptive statistics
        st.header("1. Pandas Methods")
        st.write(dfm.describe())

        # Display histograms
        st.subheader("Histograms")
        # Make the histograms look nicer (3 per column)
        dfm.hist(bins=30, figsize=(12, 8), layout=(4, -1))
        st.pyplot(plt)

        # Display kurtosis
        st.subheader("Kurtosis")
        numeric_dfm = dfm.select_dtypes(include=np.number)
        st.write(numeric_dfm.kurtosis())

        # Display skewness
        st.subheader("Skewness")
        st.write(numeric_dfm.skew())

    elif analysis_type == "Scipy Package":
        st.header("2. Scipy Package")

        # Selectbox for variables
        numeric_columns = dfm.select_dtypes(include=np.number).columns.tolist()
        if numeric_columns:
            selected_variable1 = st.selectbox("Select Variable 1:", numeric_columns)
            selected_variable2 = st.selectbox("Select Variable 2:", numeric_columns)

            y = dfm[selected_variable1]
            x = dfm[selected_variable2]

            # Function to display statistics and histogram
            def display_stats_hist(data, title):
                fig, ax = plt.subplots()
                ax.hist(data, bins='auto')
                ax.set_title(title)
                plt.tight_layout(pad = 2)
                st.pyplot(fig)
                st.write(f"-------------Statistics for {title}-------------")
                st.write("average: ", np.mean(data))
                st.write("variance  : ", np.var(data))
                st.write("skewness : ", skew(data))
                st.write("kurtosis : ", kurtosis(data))

            # Display for selected variables
            display_stats_hist(y, selected_variable1)
            display_stats_hist(x, selected_variable2)
        else:
            st.write("No numerical columns found in the CSV file.")

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

        # Correlation method selection
        correlation_method = st.selectbox("Select Correlation Method", ["pearson", "kendall", "spearman"])

        # Calculate correlation
        dfm_corr = numeric_dfm.corr(method=correlation_method)

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

        # Get numerical columns for multiselect
        numerical_columns = numeric_dfm.columns.tolist()

        # Choose variables for the pairplot
        selected_columns = st.multiselect("Select variables for pairplot", numerical_columns)

        if selected_columns:
            pairplot_fig = sns.pairplot(dfm, vars=selected_columns, hue='PRESCORING', diag_kind='hist')
            st.pyplot(pairplot_fig)

        # Heatmap
        st.write("## Heatmap")
        show_heatmap = st.checkbox("Show Heatmap", value=True)
        if show_heatmap:
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
        color: #000000;
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