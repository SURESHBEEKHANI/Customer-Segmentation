# Customer Segmentation using K-Means Clustering

This project provides a Streamlit application for performing customer segmentation using the K-Means clustering algorithm. The application is designed to be user-friendly and interactive, allowing users to explore their data and gain insights into customer behavior.

## Features

- **Interactive File Upload**: Easily upload your own CSV dataset for analysis.
- **Dynamic Feature Selection**: Select any two numerical columns for clustering.
- **Elbow Method Visualization**: Determine the optimal number of clusters using the Elbow Method.
- **Cluster Scatter Plot**: Visualize customer segments and centroids in an intuitive scatter plot.
- **Cluster Size Distribution**: View the distribution of samples across clusters with a bar chart.
- **Modular Codebase**: The project is organized into well-structured modules for utilities, clustering logic, and the Streamlit app.

## Project Structure

- `app.py`: The main Streamlit application file.
- `requirements.txt`: Contains the list of dependencies required to run the application.
- `data/`: Directory for storing datasets.
- `notebook/`: Contains Jupyter notebooks for exploratory data analysis.
- `src/`: Source code directory with the following submodules:
  - `clustering.py`: Implements the K-Means clustering logic.
  - `utils.py`: Utility functions for data preprocessing and visualization.
  - `data/Mall_Customers.csv`: Sample dataset for demonstration purposes.

## Getting Started

### Prerequisites

Ensure you have Python installed on your system. Install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

### Running the Application

To start the Streamlit application, run the following command in your terminal:

```bash
streamlit run app.py
```

### Dataset

The application works with any CSV file containing numerical data. A sample dataset (`Mall_Customers.csv`) is provided in the `src/data/` directory.

### Jupyter Notebooks

For exploratory data analysis, refer to the Jupyter notebook located in the `notebook/` directory:

- `Customer_Segmentation_using_K_Means_Clustering.ipynb`

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- The `Mall_Customers.csv` dataset is sourced from Kaggle.
- Special thanks to the open-source community for providing tools like Streamlit, Pandas, and Scikit-learn.

