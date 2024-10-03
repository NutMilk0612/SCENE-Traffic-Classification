# SCENE-Traffic-Classification
Code and experiments for SCENE: a shape-based clustering method designed for robust and noise-resilient encrypted traffic classification, optimizing service-level traffic identification across varying network conditions.
#### 1. Method Introduction

SCENE is an innovative method designed to classify encrypted traffic by utilizing network traffic behavior features combined with shape similarity. The method consists of four primary stages: traffic preprocessing, behavioral feature extraction, shape-based clustering, and statistical feature assignment. The overall architecture is shown below:

![image](https://github.com/user-attachments/assets/859d5c60-08e7-4072-b47f-f19ae87b9712)


#### 2. Quick Start Example

You can quickly obtain sample results on Dataset A by running the `StatisticalFeatureAssignment.py` file with the following command:

```
python StatisticalFeatureAssignment.py 
```

#### 3. Code File Descriptions

- `TrafficPreprocessing.py`: Implements traffic preprocessing to extract each flow from raw traffic data and generate uplink and downlink byte time series.
- `ShapeLineExtraction.py`: Extracts the shape-line from uplink and downlink traffic data, handles noise, and performs smoothing operations.
- `ShapebasedSimilarityMetric.py，DensitybasedClustering.py` : Performs density-based clustering on traffic time series using shape similarity metrics such as normalized cross-correlation (NCC).
- `StatisticalFeatureAssignment.py`: Assigns statistical features to the clustering results, calculates the central features for each cluster, and assigns anomalous flows to appropriate clusters.
- `StatisticalFeatureExtraction.py`: Contains utility functions for traffic data processing, feature computation, and other common tasks.

