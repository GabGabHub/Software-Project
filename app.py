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
import statsmodels.api as sm

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
    st.session_state.df = df 
    st.dataframe(df)

    #this is to select a column for the map
    columns = df.iloc[:, 2:].columns.tolist()
    selected_column = st.selectbox("Select a column to visualize:", columns)

    #reads map data and merges it with our dataset
    world = gpd.read_file("https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip")
    world = world.merge(df, how="left", left_on="NAME", right_on="Country")

    # generates the interactive map
    m = folium.Map(location=[20, 0], zoom_start=2)
    
    # Colors the interactive map
    def color_function(feature):
        value = feature['properties'][selected_column] if feature['properties'][selected_column] is not None else 0
        return 'green' if value > 0 else 'lightgray'  

    #this is where we actually draw the map
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

    #and this is how we put the map on the website, for some reason
    map_html = m._repr_html_()
    st.write("##Interactive map")
    st.components.v1.html(map_html, height=600)

    # doing the map the simple way
    fig, ax = plt.subplots(figsize=(10, 10))
    world.plot(column=selected_column, ax=ax, legend=True, cmap='OrRd', edgecolor='black')
    plt.title(f"Quality of Life Based on {selected_column}", fontsize=14)
    plt.legend()
    plt.show()
    st.write("Matplot Map")
    st.pyplot(fig)

def page1():
    st.title("Page 1")
    st.title("Quality of Life Statistical Analysis")  

    analysis_type = st.radio("Choose Analysis Type:", ("Pandas Methods", "Scipy Package"))
    dfm = st.session_state.df

    dfm = dfm.iloc[:, 1:]

    if analysis_type == "Pandas Methods":
        
        st.header("1. Pandas Methods")
        st.write(dfm.describe())
            
        st.subheader("Histograms")
        dfm.hist(bins=30, figsize=(12, 8), layout=(4, -1))
        plt.tight_layout()
        st.pyplot(plt)
        
        st.subheader("Kurtosis")
        numeric_dfm = dfm.select_dtypes(include=np.number)
        st.write(numeric_dfm.kurtosis())
    
        st.subheader("Skewness")
        st.write(numeric_dfm.skew())

    elif analysis_type == "Scipy Package":
        st.header("2. Scipy Package")
    
        numeric_columns = dfm.select_dtypes(include=np.number).columns.tolist()
        if numeric_columns:
            selected_variable1 = st.selectbox("Select Variable 1:", numeric_columns)
            selected_variable2 = st.selectbox("Select Variable 2:", numeric_columns)
                
            y = dfm[selected_variable1]
            x = dfm[selected_variable2]
    
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
                    
            display_stats_hist(y, selected_variable1)
            display_stats_hist(x, selected_variable2)
        else:
            st.write("No numerical columns found in the CSV file.")


def page2():
    st.title("Page 2")
    st.title("Hierarchical Cluster Analysis")

    #Preparing the data set for the HCA
    df = st.session_state.df
    df = preprocess_data(df)

    num_cols = df.select_dtypes(include=['number']).columns.drop("Rank", errors='ignore')
    df = remove_outliers_std(df, num_cols, threshold=3)

    #doing the HCA
    df = hierarchical_clustering(df)
    df, X_pca = kmeans_clustering(df)
    
    st.write("## Processed Data (After Outlier Removal)")
    st.dataframe(df)

    st.write("## K-Means Clustering Visualization")
    fig, ax = plt.subplots()
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=df['KMeans_Cluster'], cmap='viridis')
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    st.pyplot(fig)

def page3():
    st.title("Multiple Regression Model")
    
    Y = df["Quality of Life Index"]  
    X = df[["Cost of Living_Index", "Health Care Index", "Pollution Index"]]

    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    st.write(model.summary())
    
    predictions = model.predict(X)  
    df["Predicted_Quality_of_Life"] = predictions 
    st.dataframe(df[["Quality_of_Life_Index", "Predicted_Quality_of_Life"]])
    
    fig, ax = plt.subplots()
    ax.scatter(df["Quality_of_Life_Index"], df["Predicted_Quality_of_Life"], alpha=0.5)
    ax.set_xlabel("Actual Quality of Life Index")
    ax.set_ylabel("Predicted Quality of Life Index")
    ax.set_title("Actual vs Predicted Values")
    st.pyplot(fig)


if "page" not in st.session_state:
    st.session_state.page = "Home"  

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
if st.sidebar.button("HCA"):
    st.session_state.page = "HCA"
if st.sidebar.button("MRM"):
    st.session_state.page = "MRM"

# Display content based on session state
if st.session_state.page == "Home":
    home_page()
elif st.session_state.page == "Histograms":
    page1()
elif st.session_state.page == "HCA":
    page2()
elif st.session_state.page == "MRM":
    page3()
