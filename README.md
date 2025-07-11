# Machine Learning Projects (YouTube-Based)

This repository contains a collection of beginner-to-intermediate machine learning projects implemented in Python, covering a range of real-world datasets and ML techniques. These projects were completed as part of a self-learning journey through various YouTube tutorials.

Projects Overview:
File Name	Project Title	Algorithm(s) Used
-> magicml1.py:	Gamma/Hadron Classification	KNN, Naive Bayes, Logistic Regression, SVM, Neural Network
-> bikesml2.py:	Seoul Bike Demand Forecast	Linear Regression, Neural Networks (TensorFlow)
-> seedml3.py: Seed Classification	K-Means Clustering, PCA


1. Magic Gamma Telescope Classification (magicml1.py)
Dataset: magic04.data
Goal: Classify gamma and hadron particles using high-energy physics data.

Models Implemented:
>> KNN Classifier
>> Naive Bayes
>> Logistic Regression
>> Support Vector Machine
>>Custom Deep Neural Network (with dropout and tuning)

Special Features:
~ Class balancing with RandomOverSampler
~ Feature scaling using StandardScaler
~ Neural network with hyperparameter tuning: node sizes, dropout rates, batch sizes

2. Seoul Bike Demand Forecasting (bikesml2.py)
Dataset: SeoulBikeData.csv
Goal: Predict bike demand based on temperature and environmental variables.

Techniques Used:
>> Data preprocessing and feature engineering
>> Linear regression on single and multiple features
>> TensorFlow Neural Networks for regression
>> Comparison of prediction accuracy using MSE between linear regression and NN

Highlights:
~ Built multiple models: simple linear, multivariate linear, and deep neural networks
~ Trained custom TensorFlow models with different learning rates and hidden layers

3. Seed Dataset Clustering (seedml3.py)
Dataset: seeds_dataset.txt
Goal: Cluster different varieties of wheat seeds using their physical properties.

Key Concepts:
>> Pairwise plotting of features
>> Clustering with KMeans
>> Dimensionality reduction using PCA
>> Visual comparison of true labels vs. predicted clusters
>> Libraries: pandas, seaborn, matplotlib, scikit-learn

Sample Visual:
~PCA-transformed clusters colored by KMeans vs. ground truth


Learning Outcomes
-> Hands-on experience with supervised and unsupervised learning techniques
-> Gained understanding of preprocessing pipelines, model evaluation, and tuning
-> Developed visualizations for data exploration and model interpretation
-> Practiced using scikit-learn, seaborn, and TensorFlow effectively
