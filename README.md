# Laptop Market Segmentation & Recommendation Engine

This project performs an **end-to-end analysis of the Indian laptop market**, starting from raw, unstructured data and culminating in a **personalized recommendation engine**.  
It leverages **unsupervised machine learning techniques** to identify distinct market segments and provides a tool for users to find laptops that best match their specific needs and preferences.

---

## 🔑 Key Features

- **Data Cleaning & Feature Engineering**:  
  Processes messy, text-based laptop specifications into a clean, structured dataset suitable for machine learning.

- **Market Segmentation**:  
  Implements four different clustering algorithms (**K-Means, GMM, DBSCAN, Hierarchical**) to discover natural groupings of laptops based on their specs.

- **Interactive Visualization**:  
  Uses **Principal Component Analysis (PCA)** to visualize high-dimensional data in 3D, allowing for intuitive exploration of the identified clusters.

- **Personalized Recommendations**:  
  A sophisticated engine that suggests laptops based on **cosine similarity** to a user's ideal specs, including a **dynamic system for weighting the importance of each feature**.

---

## 🛠 Tech Stack

- **Python 3**
- **Pandas & NumPy** → Data manipulation and numerical operations  
- **Scikit-learn** → Feature scaling, PCA, clustering models, similarity metrics  
- **Matplotlib & Seaborn** → Static & interactive data visualizations  
- **Jupyter Notebook** → Interactive development and analysis  

---

## 📌 Project Highlights & Key Accomplishments

- Engineered **50+ features** from **995 text entries**, leveraging a **Random Forest model** for predictive data imputation.  
- Deployed **4 distinct clustering algorithms** to identify **8 key market segments**.  
- Developed **interactive 3D visualizations** by reducing **52 features → 3 principal components** using PCA.  
- Architected a **recommendation engine for 995 laptops** with a **user-controlled, dynamic feature weighting system**.  

---

## 🚀 How to Use the Recommendation Engine

1. **Define your ideal laptop specs**
   ```python
   user_specs = {
       'Price': 85000, 
       'Ram_GB': 16, 
       'SSD_GB': 512,
       'OS': 'Windows',
       'CPU': 'Intel_Core_i7', 
       'GPU': 'NVIDIA_High_Tier'
   }
feature_weights = {
    'Price': 10,
    'Ram_GB': 8,
    'GPU_NVIDIA_High_Tier': 9
}
recommendations = recommend_laptops_with_weights(
    specs=user_specs, 
    df=df1, 
    weights=feature_weights, 
    n_rec=10
)

print("Your top 10 personalized recommendations:")
display(recommendations)
