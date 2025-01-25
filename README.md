# Customer Personality Analysis
 Here's a concise and structured description for your GitHub repository:

---

# **Customer Personality Analysis**

## **Overview**  
This project focuses on analyzing customer personality traits to help businesses understand their customer base better. By segmenting customers into distinct groups using clustering techniques, businesses can optimize their marketing strategies, customize product offerings, and improve customer satisfaction.

---

## **Dataset Description**  
The dataset contains detailed information about customer demographics, spending habits, and engagement with promotional campaigns. The data is categorized as follows:  

### **1. People**  
- **ID**: Unique identifier for each customer.  
- **Year_Birth**: Customer's year of birth.  
- **Education**: Education level.  
- **Marital_Status**: Marital status.  
- **Income**: Yearly household income.  
- **Kidhome/Teenhome**: Number of children and teenagers in the household.  
- **Dt_Customer**: Enrollment date.  
- **Recency**: Days since last purchase.  
- **Complain**: Whether the customer complained in the last 2 years.  

### **2. Products**  
- Spending on **wine, fruits, meat, fish, sweets, and gold** in the last 2 years.  

### **3. Promotion**  
- Engagement with promotional campaigns (e.g., discounts, special offers).  

### **4. Place**  
- Purchases made via website, catalog, or store, along with website visit frequency.

---

## **Project Goals**  
- Perform **data preprocessing** (handling missing values, encoding categorical variables, scaling numerical features).  
- Apply **clustering algorithms** to segment customers.  
- Evaluate clustering performance using metrics such as **Silhouette Score**, **Calinski-Harabasz Index**, and **Davies-Bouldin Index**.  
- Use **Principal Component Analysis (PCA)** for dimensionality reduction to enhance clustering results and visualization.

---

## **Technologies Used**  
- Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)  
- Clustering Algorithms (K-Means, DBSCAN, Hierarchical Clustering, Gaussian Mixture Models)  
- Dimensionality Reduction (PCA)

---

## **Methodology**  

### **1. Data Preprocessing**  
- Imputed missing values (e.g., income).  
- Encoded categorical variables and standardized numerical features.  
- Removed irrelevant and constant features.

### **2. Clustering Techniques**  
- **K-Means Clustering**: Optimal clusters determined using the elbow method.  
  - Silhouette Score: 0.598  
  - Calinski-Harabasz Index: 9028.17  
  - Davies-Bouldin Index: 0.495  
- **PCA**: Reduced dimensions for better visualization and efficiency.  
- **Other Methods**: Hierarchical clustering, DBSCAN, and Gaussian Mixture Models for comparison.

### **3. Insights**  
- Segmentation reveals distinct customer groups based on demographics, spending habits, and promotional engagement.  
- PCA improved visualization and efficiency while retaining key information.

---





