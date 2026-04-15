# 🛍️ Customer Segmentation using K-Means Clustering

##  Project Overview

This project performs **customer segmentation** for a retail dataset using **K-Means Clustering**.
The goal is to group customers based on their **income and spending behavior** to enable **targeted marketing strategies**.

---

##  Objectives

* Identify distinct customer segments
* Improve marketing decision-making
* Understand customer behavior patterns
* Provide actionable business insights

---

## Project Structure

```
mall_customer_segmentation/
│
├── data/
│   └── Mall_Customers.csv
│
├── outputs/
│   ├── customer_segmentation_final.png
│   ├── pca_cluster_visualization_final.png
│   ├── cluster_summary_final.csv
│   └── customer_segments_final.csv
│
├── customer_segmentation.py
├── requirements.txt
└── README.md
```

---

## Dataset Information

* **Rows:** 200 customers
* **Features:**

  * CustomerID
  * Gender
  * Age
  * Annual Income (k$)
  * Spending Score (1–100)

---

## Methodology

### 1️⃣ Feature Selection

Used:

* Annual Income (k$)
* Spending Score (1–100)

 Chosen to ensure **no redundancy and high interpretability**

---

### 2️⃣ Data Scaling

* Applied **MinMaxScaler**
* Ensures equal contribution of features to distance-based clustering

---

### 3️⃣ Optimal K Selection

Evaluated using:

* Elbow Method
* Silhouette Score
* Davies-Bouldin Index

 Final selection:

* **k = 5 clusters**

---

### 4️⃣ Model Training

* Algorithm: **K-Means Clustering**
* Initialization: `random_state=42`
* Scaled feature space used

---

### 5️⃣ Evaluation Metrics

| Metric               | Value  | Interpretation          |
| -------------------- | ------ | ----------------------- |
| Silhouette Score     | 0.5595 | Good cluster separation |
| Davies-Bouldin Index | 0.5678 | Low overlap             |
| Calinski-Harabasz    | 264.73 | Strong structure        |

---

### 6️⃣ PCA Visualization

* Applied **Principal Component Analysis (PCA)**
* Reduced to 2D for visualization
* **100% variance retained** (due to 2 features)

---

## Results

### Customer Segments Identified:

*  Premium Spenders
* Conservative Shoppers
* Aspirational Shoppers
* Budget Conscious
* Average Customers

---

## Visualizations

* Elbow & Silhouette Analysis
* Davies-Bouldin Index Plot
* Cluster Scatter Plot
* Segment Distribution
* PCA Visualization

---

## Business Insights

### Targeted Strategies:

* **Premium Spenders**

  * Luxury products
  * VIP memberships
  * Exclusive offers

* **Aspirational Shoppers**

  * Discounts & EMI options
  * Personalized promotions

* **Conservative Shoppers**

  * Savings-focused campaigns
  * Investment-related products

* **Budget Conscious**

  * Coupons & value deals
  * Bundle offers

* **Average Customers**

  * General marketing campaigns
  * Loyalty programs

---

## Key Takeaways

* Premium customers provide highest revenue potential
* Conservative customers represent untapped opportunity
* Aspirational customers show strong brand growth potential
* Largest segment (Average Customers) drives volume

---

## 🚀 How to Run

### 1️⃣ Clone the repository

```
git clone <your-repo-link>
cd mall_customer_segmentation
```

### 2️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 3️⃣ Run the script

```
python customer_segmentation.py
```

---

## Requirements

```
numpy==1.26.4
pandas==2.2.2
matplotlib==3.8.4
scikit-learn==1.4.2
scipy==1.13.1
```

---

## Features of This Project

* ✔ Clean feature selection
* ✔ Multiple evaluation metrics
* ✔ PCA visualization
* ✔ Business-oriented segmentation
* ✔ Reproducible results

---

## Future Improvements

* Add more features (e.g., purchase history, frequency)
* Try other clustering algorithms (DBSCAN, Hierarchical)
* Deploy as a web app/dashboard

---

## Author

**Diya Benny**

---
