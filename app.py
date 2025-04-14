import streamlit as st
import matplotlib.pyplot as plt
from src.clustering import load_data, extract_features, fit_kmeans, calculate_wcss
from src.utils import plot_cluster_counts, visualize_clusters
from typing import List

# Page configuration
st.set_page_config(
    page_title="Customer Segmentation",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar styling via markdown (optional)
st.markdown(
    """
    <style>
    .reportview-container { padding: 2rem; }
    .sidebar .sidebar-content { background-color: #ffffff; padding: 1.5rem; border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True
)

# App title
st.title("📊 Customer Segmentation")

# File upload
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
if not uploaded_file:
    st.sidebar.info("Please upload a CSV file to proceed.")
    st.stop()

# Load data
data = load_data(uploaded_file)
st.subheader("Dataset Preview")
st.dataframe(data.head())

# Select features
feature_options: List[str] = list(data.columns)
selected_features = st.sidebar.multiselect(
    "Select two features for clustering:",
    options=feature_options,
    default=feature_options[3:5]
)
if len(selected_features) != 2:
    st.sidebar.error("Please select exactly two features.")
    st.stop()

# Clustering settings
n_clusters = st.sidebar.slider(
    "Number of clusters", min_value=2, max_value=10, value=5
)

# Run clustering
if st.sidebar.button("Run Clustering"):
    # Extract features
    X = extract_features(data, selected_features)

    # Compute elbow
    wcss = calculate_wcss(X, max_clusters=10)
    fig_elbow, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(wcss) + 1), wcss, marker='o')
    ax.set_title("Elbow Method: WCSS vs. Number of Clusters", fontsize=14)
    ax.set_xlabel("Number of Clusters", fontsize=12)
    ax.set_ylabel("WCSS", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    st.subheader("Elbow Method")
    st.pyplot(fig_elbow)

    # Fit KMeans
    labels, centers = fit_kmeans(X, n_clusters)
    data['Cluster'] = labels

    # Cluster visualization
    st.subheader("Cluster Plot")
    fig_clusters = visualize_clusters(X, labels, centers)
    st.pyplot(fig_clusters)

    # Cluster counts
    st.subheader("Cluster Size Distribution")
    fig_counts = plot_cluster_counts(labels)
    st.pyplot(fig_counts)

    st.success("Clustering completed!")