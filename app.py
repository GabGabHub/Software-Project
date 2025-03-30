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
import geopandas as gpd
import folium

def remove_outliers_std(df, num_cols, threshold=3):
    z_scores = np.abs((df[num_cols] - df[num_cols].mean()) / df[num_cols].std())
    return df[(z_scores < threshold).all(axis=1)]

def preprocess_data(df):
    df = df.fillna(0)
    num_cols = df.select_dtypes(include=['number']).columns
    st.write(num_cols)
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
                
    world = gpd.read_file("https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip")
    world = world.merge(df, how="left", left_on="NAME", right_on="Country")
    
    m = folium.Map(location=[20, 0], zoom_start=2)
    
    # Create a function to color countries based on the selected column
    def color_function(feature):
        value = feature['properties'][selected_column] if feature['properties'][selected_column] is not None else 0
        return 'green' if value > 0 else 'lightgray'  # Adjust colors based on your needs

    folium.GeoJson(
        world,
        style_function=lambda feature: {
            'fillColor': color_function(feature),
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.6,
        },
        name='World Countries'
    ).add_to(m)

    bounds = [[-60, -180], [85, 180]]
    m.fit_bounds(bounds)
    map_html = m._repr_html_()
    st.components.v1.html(map_html, height=600)

    # Plotting with Matplotlib
    fig, ax = plt.subplots(figsize=(10, 10))
    world.plot(column=selected_column, ax=ax, legend=True, cmap='OrRd', edgecolor='black')
    plt.title(f"Quality of Life Based on {selected_column}", fontsize=14)
    plt.legend()
    plt.show()
    st.pyplot(fig)

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
    st.title("Clustering Analysis")

    # Access the DataFrame from session state
    if "df" in st.session_state:
        df = st.session_state.df
        df = preprocess_data(df)  # Standardization

        # Remove outliers using standard deviation method
        num_cols = df.select_dtypes(include=['number']).columns.drop("Rank", errors='ignore')
        df = remove_outliers_std(df, num_cols, threshold=3)

        # Apply clustering
        df = hierarchical_clustering(df)
        df, X_pca = kmeans_clustering(df)

        st.write("## Processed Data (After Outlier Removal)")
        st.dataframe(df.head())

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
