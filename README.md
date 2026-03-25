# Compute Distances in the Iris Dataset

A Python implementation that computes three classical distance metrics — **Standardized Euclidean**, **Cosine**, and **Mahalanobis** — across all 150 samples in the Iris dataset, visualizes the full pairwise distance matrices, and then derives compact **3×3 inter-class proximity matrices** to quantify how separated the three Iris species are under each metric.

---

## Overview

Distance metrics are foundational to machine learning algorithms like k-Nearest Neighbors and clustering. This project goes beyond simply computing pairwise distances — it also builds per-class proximity summaries, making it easy to see which metric best separates the three Iris species from one another.

---

## Repository Structure

```
Compute-Distances-in-iris-dataset/
├── Compute Distances     # Core Python script
├── iris_1.xlsx           # Iris dataset (Excel format)
└── README.md
```

---

## Dataset

`iris_1.xlsx` contains the classic [Iris flower dataset](https://archive.ics.uci.edu/ml/datasets/iris) with **150 samples** across three species — *Iris setosa* (rows 1–50), *Iris versicolor* (rows 51–100), and *Iris virginica* (rows 101–150) — each described by four numeric features:

| Feature | Description |
|---------|-------------|
| `sepal_length` | Length of the sepal (cm) |
| `sepal_width` | Width of the sepal (cm) |
| `petal_length` | Length of the petal (cm) |
| `petal_width` | Width of the petal (cm) |

> The script drops the `FN` label column before processing, leaving a pure 150×4 numeric matrix.

---

## Pipeline

### Step 1 — Load & Preprocess
The Excel file is read with `pandas`, the non-numeric `FN` column is removed, and the data is converted to a NumPy array. The matrix is then **column-normalized** using `sklearn.preprocessing.normalize` (axis=0), so each feature is scaled to unit norm before any distances are computed.

```python
df_norm = normalize(df, axis=0)
```

---

### Step 2 — Compute Distance Matrices

All three pairwise distance matrices are computed using `scipy.spatial.distance.cdist`, producing **150×150 symmetric matrices**.

#### Standardized Euclidean Distance
A weighted form of Euclidean distance that accounts for differences in variance across features. Each squared difference is divided by the variance of that feature:

```
d(x, y) = sqrt( Σ ((xᵢ - yᵢ)² / Vᵢ) )
```

where **V** is the per-feature variance vector computed from the data.

```python
df_eucl = distance.cdist(df_norm, df_norm, 'seuclidean')
```

#### Cosine Distance
Measures the angle between two sample vectors, ignoring magnitude and focusing purely on orientation:

```
cosine_distance = 1 - (x · y) / (‖x‖ × ‖y‖)
```

A value of 0 means the vectors point in the same direction; 1 means they are orthogonal.

```python
df_cosine = distance.cdist(df_norm, df_norm, 'cosine')
```

#### Mahalanobis Distance
Accounts for correlations between features and differences in scale. It transforms the data using the inverse covariance matrix before measuring distance:

```
d(x, y) = sqrt( (x - y)ᵀ × S⁻¹ × (x - y) )
```

where **S** is the covariance matrix of the full dataset. This makes it scale-invariant and robust to correlated features.

```python
df_maha = distance.cdist(df_norm, df_norm, 'mahalanobis')
```

---

### Step 3 — Visualize Distance Matrices

Each 150×150 matrix is displayed as a heatmap using `matplotlib.pyplot.matshow`, making it visually clear how tightly grouped or spread out each species is under each metric.

```python
plt.matshow(df_eucl)
plt.matshow(df_cosine)
plt.matshow(df_maha)
```

The three diagonal blocks (rows/columns 0–49, 50–99, 100–149) correspond to intra-species distances. A well-separating metric shows dark diagonal blocks and bright off-diagonal regions.

---

### Step 4 — Build Proximity Matrices

For each distance metric, the 150×150 matrix is split into the three known species blocks and the **mean intra-class distance (centroid)** is computed for each:

```python
# Example for Euclidean
df_eucl_split_1 = df_eucl[0:50,   0:50]    # setosa vs setosa
df_eucl_split_2 = df_eucl[50:100, 50:100]  # versicolor vs versicolor
df_eucl_split_3 = df_eucl[100:150,100:150] # virginica vs virginica
```

These centroids are used to build a **3×3 proximity matrix** where:
- **Diagonal entries** = mean intra-class distance (compactness of each species)
- **Off-diagonal entries** = difference between class centroids (inter-class separation)

```
proximity[i, j] = centroid_j - centroid_i
```

This results in three 3×3 matrices — one per distance metric — that compactly summarize how well each metric separates the three Iris species.

---

## Distance Metric Comparison

| Metric | Variant Used | Scale-Invariant | Correlation-Aware | Best For |
|--------|-------------|-----------------|-------------------|----------|
| Euclidean | Standardized (`seuclidean`) | Yes (variance-weighted) | No | Independent features with differing variance |
| Cosine | Standard | Yes (magnitude-independent) | No | Comparing feature proportions/directions |
| Mahalanobis | Standard | Yes | Yes | Correlated features; detecting multivariate outliers |

---
