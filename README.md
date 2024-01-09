# Computing Science Bachelor Thesis #
* Paper: Utilizing machine learning and SHAP to uncover key variables for a healthier lifestyle in Type 2 Diabetes management
* Python files:
    * machineLearning.py:
        * Uses SQL statements to extract data from Microsoft Azure SQL Databases
        * Uses GET requests to extract data from Hubspot
        * Creates a combined Dataframe using an inner-join
        * Uses feature engineering and data preprocessing to prepare data 
        * Uses sklearn library to execute 6 machine learning algorithms
        * Uses evaluation metrics such as: classification reports, confusion matrices, ROC-curves and K-fold accuracy scores
        * Uses KernalExplainer from shap library as an interpretability tool
    * metingen.py:
        * Uses SQL statements to extract data from Microsoft Azure SQL Databases
        * Creates Graphs that take the average difference of HbA1c, weight, glucose and WtHr over a period of 30-months    
    * onlineActiviteit.py: 
        * Uses GET requests to extract data from Hubspot
        * Uses bar charts to display to amount of online activity against measurement results
* Website:
    * Uses html+css templates for two pages
    * app.py uses functions from the python files to display the average difference of HbA1c, weight, glucose and waist circumference from 2019 - present
