### Data Analysis and Machine Learning Algorithm Analysis

This project encompasses the applications of data analysis and machine learning algorithms. The project performs various operations on two different datasets. The first dataset contains features affecting Student Stress Factors. The aim with this dataset is to analyze student stress factors and predict sleep quality. The second dataset contains information about employees' jobs and salaries. The objective with this dataset is to perform job clustering and salary prediction.


  - [Datasets](#Datasets)
  - [Data Analysis and Cleaning](#DataAnalysisandCleaning)
  - [Regression Analysis](#RegressionAnalysis)
  - [Dimensionality Reduction](#DimensionalityReduction)
  - [Clustering Analysis](#ClusteringAnalysis)
  - [Prediction Algorithms](#PredictionAlgorithms)
  - [Results](#Results)
  - [How to Run](#HowtoRun)
  - [Resources](#Resources)

###  Datasets

The Student Stress Factors dataset contains approximately 20 features that have the most significant impact on a student's stress. These features are categorized into 5 main categories: `Psychological`, `Physiological`, `Social`, `Environmental`, and `Academic Factors`. Some psychological factors include attributes like 'anxiety_level', 'self-esteem', 'mental_health_history', and 'depression'.

With this dataset, an attempt has been made to analyze student stress factors and predict sleep quality.

The Jobs and Salaries Data consists of information regarding the jobs and salaries of 5000 employees. Factors such as job title, job description, industry sector, workplace, working hours, education level, years of experience, skills, salary, and salary range are included in this dataset. The aim with this dataset is to perform job clustering and salary prediction.

### Data Analysis and Cleaning

For each dataset, features have been analyzed, and statistics have been extracted. The distributions, correlations, outliers, and missing values of the datasets have been examined.
The datasets have been cleaned, and missing data have been filled. Unnecessary columns have been removed, and the data has been prepared for training. The final versions of the datasets have been saved as **cleaned_mobile.csv** and **cleaned_jobs.csv**.


### Regression Analysis
- A linear regression model has been applied to the Student Stress Factors dataset.
- Psychology has been selected as the dependent variable, while other features are considered independent variables.
- The R^2 value of the linear regression model has been determined to be 0.92.
- The coefficients and significance values of the model are presented in the **linear_regression_results.py** file.
- Logistic regression classification models have been developed to understand the relationship between stress and sleep quality.
- The first logistic regression model binary classifies sleep quality as good or poor.
- The second logistic regression model classifies sleep quality as very good, good, fair, poor, or very poor.
- The accuracy, precision, recall, and F1 scores of the models are reported in the **logistic_regression_results.py** file.

  
### Dimensionality Reduction
- PCA dimensionality reduction method has been applied to the data. The optimal number of dimensions has been determined for each dataset, and the data has been reduced to this number of dimensions. After dimensionality reduction, the resulting data has been saved as pca_mobile.csv and pca_jobs.csv.

### Clustering Analysis

1. **K-Means and GMM Clustering Methods Have Been Applied:**
   - K-Means and GMM clustering methods have been applied to the jobs and salaries data.

2. **The Optimal Number of Clusters Has Been Determined:**
   - The optimal number of clusters has been determined for each method.

3. **Clustering Results Have Been Saved:**
   - The results of K-Means and GMM clustering methods have been saved separately.
   - The results have been saved as `kmeans_jobs.csv` and `gmm_jobs.csv` files.

4. **Jobs Characteristics and Salaries Grouping Have Been Examined:**
   - Based on the clustering results, an examination has been conducted on how jobs' characteristics and salaries are grouped.

5. **Common Characteristics, Differences, and Distributions of Clusters Have Been Presented:**
   - In the `clustering_results.py` file, common characteristics, differences, and distributions of clusters have been presented.
  
     
### Prediction Algorithms

- **Algorithm Development and Data Splitting:**
  - XGB and artificial neural network algorithms have been developed for salary prediction.
  - The data has been split into training, validation, and test sets.
  - The hyperparameters of the algorithms have been optimized.
  - The purpose and operation of the algorithms are explained in the **prediction_algorithms.txt** file.

- **Performance Metrics:**
  - The performance of the algorithms has been evaluated using metrics such as mean absolute error, mean squared error, R^2 value, and other metrics.
  - Performance metrics are detailed in the **prediction_results.py** file.
  - The performances of the algorithms have been compared.

- **Visualization:**
  - The obtained results have been visualized using Matplotlib or Seaborn libraries.
  - The visuals are stored in the **figures** folder.
  - Each visual's name specifies which process or result it relates to.
  - For example, the **linear_regression_plot.py** file displays the output of the linear regression model.
 
  
 ###  Results

<img width="400" alt="Ekran Resmi 2024-02-18 17 35 25" src="https://github.com/ogulcandeniz-inac/MachineLearningStressAnalysis/assets/109241786/2a8784d7-59b9-4f47-b7f2-1818a93f72f1">
<img width="400" alt="Ekran Resmi 2024-02-18 17 35 19" src="https://github.com/ogulcandeniz-inac/MachineLearningStressAnalysis/assets/109241786/8e511503-b488-4b35-a96e-763bc417408b">
<img width="400" alt="Ekran Resmi 2024-02-18 17 38 00" src="https://github.com/ogulcandeniz-inac/MachineLearningStressAnalysis/assets/109241786/9b4abf2a-f880-4d5f-a52e-423f7242f29d">
<img width="400" alt="Ekran Resmi 2024-02-18 17 38 12" src="https://github.com/ogulcandeniz-inac/MachineLearningStressAnalysis/assets/109241786/97eb67ee-2b94-4d4e-a4e8-17d3247440e2">



 
    
### How to Run
- Python 3.8 or higher version is required.
- Required libraries include: pandas, numpy, scikit-learn, xgboost, tensorflow, matplotlib, and seaborn.
- You can use the `requirements.py` file to install the libraries.
- To install the libraries, run the command `pip install -r requirements.py` in the command line.
- After downloading the datasets and code files, run the `main.py` file to initiate the project.
- `main.py` sequentially calls all the operations performed in the project.
- To initiate the project, use the command `python main.py` in the command line.
- You can review the outputs of the code in the `results` folder.
- In the `results` folder, you can find performance metrics of created models, clustering results, prediction algorithms, and visualizations.
- Contribution steps:
  1. Fork or clone the project from GitHub.
  2. Create a new branch and make your changes.
  3. Commit and push your changes.
  4. Create a pull request and submit it to the project owner.
  5. Wait for your pull request to be reviewed and accepted.


### Resources
The sources used or referenced within the scope of the project are as follows:

- Student Stress Factors Dataset: [https://www.kaggle.com/datasets/rxnach/student-stress-factors-a-comprehensive-analysis]
- Jobs and Salaries Data: [https://www.kaggle.com/airiddha/trainrev1]
- Linear Regression: [https://scikit-learn.org/stable/modules/linear_model.html]
- Logistic Regression: [https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression]
- PCA: [https://scikit-learn.org/stable/modules/decomposition.html#pca]
- K-Means: [https://scikit-learn.org/stable/modules/clustering.html#k-means]
- GMM: [https://scikit-learn.org/stable/modules/mixture.html#gmm]
- XGB: [https://xgboost.readthedocs.io/en/latest/]
- Artificial Neural Network: [https://www.tensorflow.org/tutorials/keras/regression]
- Matplotlib: [https://matplotlib.org/]
- Seaborn: [https://seaborn.pydata.org/]
