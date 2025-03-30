import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kurtosis, skew
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def preprocess_data(df):
    df = df.fillna(0)
    num_cols = df.select_dtypes(include=['number']).columns
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df


def hierarchical_clustering(df):
    num_cols = df.select_dtypes(include=['number']).columns
    X = df[num_cols].values

    HC = linkage(X, method='ward')

    def threshold(h):
        n = h.shape[0]
        dist_1 = h[1:n, 2]
        dist_2 = h[0:n - 1, 2]
        diff = dist_1 - dist_2
        j = np.argmax(diff)
        t = (h[j, 2] + h[j + 1, 2]) / 2
        return t, j, n

    t, j, n = threshold(HC)
    k = n - j
    labels = fcluster(HC, k, criterion='maxclust')
    df['Hierarchical_Cluster'] = labels
    return df


def correlation_analysis(df):
    num_cols = df.select_dtypes(include=['number'])
    st.write(num_cols)
    st.write('## WHAT THE SIGMA')
    correlation_method = st.selectbox("Select Correlation Method", ["pearson", "kendall", "spearman"])
    df_corr = df[num_cols].corr(method=correlation_method)

    st.write("## Correlation Matrix")
    st.write(df_corr)

    st.write("## Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df_corr, annot=True, ax=ax)
    st.pyplot(fig)

def kmeans_clustering(df):
    num_cols = df.select_dtypes(include=['number']).columns
    X = df[num_cols].values

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    kmeans = KMeans(n_clusters=5, n_init=10)
    df['KMeans_Cluster'] = kmeans.fit_predict(X_pca)
    return df, X_pca

def home_page():
    st.title("Database")
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
        plt.tight_layout()
        # plt.subplots_adjust(hspace=0.5, wspace=0.5)
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
        df = st.session_state.df
        df = preprocess_data(df)
        df = hierarchical_clustering(df)
        df, X_pca = kmeans_clustering(df)

        st.write("## Processed Data")
        st.dataframe(df.head())

        correlation_analysis(df)

        st.write("## K-Means Clustering Visualization")
        fig, ax = plt.subplots()
        ax.scatter(X_pca[:, 0], X_pca[:, 1], c=df['KMeans_Cluster'], cmap='viridis')
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        st.pyplot(fig)

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
