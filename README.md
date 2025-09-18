# Laptop-Market-Segmentation-and-Recommendation-Engine-
Laptop Market Segmentation & Recommendation Engine
This project performs an end-to-end analysis of the Indian laptop market, starting from raw, unstructured data and culminating in a personalized recommendation engine. It leverages unsupervised machine learning techniques to identify distinct market segments and provides a tool for users to find laptops that best match their specific needs and preferences.

Key Features
Data Cleaning & Feature Engineering: Processes messy, text-based laptop specifications into a clean, structured dataset suitable for machine learning.

Market Segmentation: Implements four different clustering algorithms (K-Means, GMM, DBSCAN, Hierarchical) to discover natural groupings of laptops based on their specs.

Interactive Visualization: Uses Principal Component Analysis (PCA) to visualize high-dimensional data in 3D, allowing for intuitive exploration of the identified clusters.

Personalized Recommendations: A sophisticated engine that suggests laptops based on cosine similarity to a user's ideal specs, including a dynamic system for weighting the importance of each feature.

Tech Stack
Python 3

Pandas & NumPy: For data manipulation and numerical operations.

Scikit-learn: For feature scaling, PCA, clustering models, and similarity metrics.

Matplotlib & Seaborn: For static and interactive data visualizations.

Jupyter Notebook: For interactive development and analysis.

Project Highlights & Key Accomplishments
Engineered over 50 features from 995 text entries, leveraging a Random Forest model for predictive data imputation.

Deployed 4 distinct clustering algorithms (K-Means, GMM, DBSCAN, Hierarchical) to identify 8 key market segments.

Developed interactive 3D visualizations by reducing 52 features to 3 principal components using PCA for cluster analysis.

Architected a recommendation engine for 995 laptops with a user-controlled, dynamic feature weighting system.

How to Use the Recommendation Engine
To get personalized recommendations, define your desired specifications and feature importance in dictionaries, then call the main function.

# 1. Define your ideal laptop specs
user_specs = {
    'Price': 85000, 
    'Ram_GB': 16, 
    'SSD_GB': 512,
    'OS': 'Windows',
    'CPU': 'Intel_Core_i7', 
    'GPU': 'NVIDIA_High_Tier'
}

# 2. Define the importance of each feature (1-10 scale)
feature_weights = {
    'Price': 10,
    'Ram_GB': 8,
    'GPU_NVIDIA_High_Tier': 9
}

# 3. Get personalized recommendations
recommendations = recommend_laptops_with_weights(
    specs=user_specs, 
    df=df1, 
    weights=feature_weights, 
    n_rec=10
)

print("Your top 10 personalized recommendations:")
display(recommendations)

Project Files
cleaning and clustering.ipynb: The main Jupyter Notebook containing all data processing, clustering analysis, and the recommendation engine code.
