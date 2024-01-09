import pandas as pd
import pypyodbc as odbc
import requests
from credentials import conn_string, API_key
from sklearn.model_selection import GridSearchCV, train_test_split, KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

shap.initjs()

ALL_ENDPOINT = 'https://api.hubapi.com/contacts/v1/lists/4/contacts/all?count=300' # set count to the number of members in the list
PROFILE_ENDPOINT = 'https://api.hubapi.com/contacts/v1/contact/vid/'

mtypeid = [('3', ['last_waist_date', 'last_waist', 'first_waist_date', 'first_waist', 'waist_duration']), ('4', ['last_glucose_date', 'last_glucose', 'first_glucose_date', 'first_glucose', 'glucose_duration']), ('5', ['last_HbA1c_date', 'last_HbA1c', 'first_HbA1c_date', 'first_HbA1c', 'HbA1c_duration'])]

# Take weight measurements of every person 
sql1 = '''
    SELECT M.personid, M.mtypeid, M.measurementdate, M.measuredvalue, P.voornaam AS first_name, P.achternaam AS last_name, P.geslacht AS gender, P.lengte AS length, P.geboortedatum AS birthdate  
    FROM dbo.Measurements AS M
    INNER JOIN dbo.Personen AS P
    ON M.personid=P.id
    WHERE M.mtypeid=?
    ORDER BY M.measurementdate
    '''

sql2 = '''
    SELECT M.personid, M.measurementdate, M.measuredvalue, P.voornaam AS first_name, P.achternaam AS last_name
    FROM dbo.Measurements AS M
    INNER JOIN dbo.Personen AS P
    ON M.personid=P.id
    WHERE M.mtypeid=?
    ORDER BY M.measurementdate
    '''

# Take first and last measurement
def firstAndLastDate(df, isWeight, names):
    df['measurementdate'] = pd.to_datetime(df['measurementdate'])

    first_measurement_idx = df.groupby('personid')['measurementdate'].idxmin()
    last_measurement_idx = df.groupby('personid')['measurementdate'].idxmax()
    
    if isWeight:
        first_last_measurements = df.loc[first_measurement_idx, ['personid', 'first_name', 'last_name', 'gender', 'length', 'birthdate', 'measurementdate', 'measuredvalue']]
    else:
        first_last_measurements = df.loc[first_measurement_idx, ['first_name', 'last_name', 'measurementdate', 'measuredvalue']]
        
    first_last_measurements[names[0]] = df.loc[last_measurement_idx, 'measurementdate'].values
    first_last_measurements[names[1]] = df.loc[last_measurement_idx, 'measuredvalue'].values
    first_last_measurements.rename(columns={'measurementdate': names[2], 'measuredvalue': names[3]}, inplace=True)
    first_last_measurements[names[4]] = (first_last_measurements[names[0]] - first_last_measurements[names[2]]).astype(str).str.extract('(\d+)').astype(int)
    
    return first_last_measurements

# Change weight to BMI
def filteredWeight(dfWeight):
    names = ['last_weight_date', 'last_weight', 'first_weight_date', 'first_weight', 'weight_duration']
    newDf = firstAndLastDate(dfWeight, True, names)
    # convert birth date to age rounded down
    newDf['birthdate'] = pd.to_datetime(newDf['birthdate'])
    current_date = pd.to_datetime('today')
    newDf['current_age'] = np.floor(((current_date - newDf['birthdate']).dt.days) / 365.25)
    # add first and last BMI
    newDf['first_BMI'] = newDf['first_weight'] / ((newDf['length']/100)*(newDf['length']/100))
    newDf['last_BMI'] = newDf['last_weight'] / ((newDf['length']/100)*(newDf['length']/100))
    return newDf

# T2D classification function
def diabetes(df_row):
    points = 0
    if df_row['gender'] == 'M':
        points += 1

    if df_row['age'] >= 50 and df_row['age'] < 60:
        points += 4
    elif df_row['age'] >= 60 and df_row['age'] < 70:
        points += 6
    elif df_row['age'] >= 70:
        points += 9

    if df_row['waist'] >= 90 and df_row['waist'] < 100:
        points += 4
    elif df_row['waist'] >= 100 and df_row['waist'] < 110:
        points += 6
    elif df_row['waist'] >= 110:
        points += 9
    
    if df_row['BMI'] >= 25 and df_row['BMI'] < 30:
        points += 4
    elif df_row['BMI'] >= 30 and df_row['BMI'] < 35:
        points += 6
    elif df_row['BMI'] >= 35:
        points += 9

    if df_row['glucose'] >= 5.6 and df_row['glucose'] < 6.1:
        points += 4
    if df_row['glucose'] >= 6.1 and df_row['glucose'] < 7:
        points += 6
    elif df_row['glucose'] >= 7:
        points += 9

    if df_row['HbA1c'] >= 42 and df_row['HbA1c'] < 49:
        points += 4
    elif df_row['HbA1c'] >= 49 and df_row['HbA1c'] < 53:
        points += 6
    elif df_row['HbA1c'] >= 53:
        points += 9

    if points >= 31:
        return 1
    else:
        return 0

# GET REQUEST to Hubspot
def get_request(endpoint):
    headers = {
        'Authorization': f'Bearer {API_key}'
    }
    response = requests.get(endpoint, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f'Error: {response.status_code}')

# Assemble all canonical-vids from the right Hubspot list 
def get_all_contacts_vids():
    data = get_request(ALL_ENDPOINT)
    canonical_vids = [contact['canonical-vid'] for contact in data['contacts']]
    return canonical_vids

# GET name + online activity of each member in a Dataframe
def get_individual_page():
    canonical_vids = get_all_contacts_vids()
    df = pd.DataFrame(columns=['first_name', 'last_name', 'num_visits', 'num_views'])
    index = 0
    for vid in canonical_vids:
        newEndpoint =  PROFILE_ENDPOINT + str(vid) + '/profile?propertyMode=value_only'
        data = get_request(newEndpoint)
        first_name = data.get('properties', {}).get('firstname', {}).get('value', None)
        last_name = data.get('properties', {}).get('lastname', {}).get('value', None)
        num_visits = data.get('properties', {}).get('hs_analytics_num_visits', {}).get('value', None)
        num_page_views = data.get('properties', {}).get('hs_analytics_num_page_views', {}).get('value', None)
        df.loc[index] = [first_name, last_name, num_visits, num_page_views]
        index = index+1
    return df

# Data from ZWEM
def retrieveZWEMData(mtypeid, isWeight):
    conn = odbc.connect(conn_string)

    # Execute SQL statement and store in DataFrame
    cursor = conn.cursor()
    if isWeight:
        cursor.execute(sql1, (mtypeid,))
    else:
        cursor.execute(sql2, (mtypeid,))

    dataset = cursor.fetchall()
    columns = [column[0] for column in cursor.description] 
    
    return pd.DataFrame(dataset, columns=columns)

def first_age(row):
    return np.floor(row['current_age'] - (row['weight_duration']/365.25))

def waistToLengthRatioConverterFirst(row):
    return (row['first_waist'] / row['length'])

def waistToLengthRatioConverterLast(row):
    return (row['last_waist'] / row['length'])

def featureEngineering(df):
    df['first_wthr'] = df.apply(waistToLengthRatioConverterFirst, axis=1)
    df['last_wthr'] = df.apply(waistToLengthRatioConverterLast, axis=1)
    df['first_age'] = df.apply(first_age, axis=1)

def splitFirstandLastMeasurement(df):
    first_measurement_df = df[['first_name', 'last_name', 'gender', 'first_age', 'first_BMI', 'first_wthr', 'first_waist', 'first_glucose', 'first_HbA1c']].copy()
    first_measurement_df['num_visits'] = 0 # add zero values for first measurement
    first_measurement_df['num_views'] = 0 
    last_measurement_df = df[['first_name', 'last_name', 'gender', 'current_age', 'last_BMI', 'last_wthr', 'last_waist', 'last_glucose', 'last_HbA1c', 'num_visits', 'num_views']].copy()

    first_measurement_df.columns = ['first_name', 'last_name', 'gender', 'age', 'BMI', 'wthr', 'waist', 'glucose', 'HbA1c', 'num_visits', 'num_views']
    last_measurement_df.columns = ['first_name', 'last_name', 'gender', 'age', 'BMI', 'wthr', 'waist', 'glucose', 'HbA1c', 'num_visits', 'num_views']

    return pd.concat([first_measurement_df, last_measurement_df], ignore_index=True)

# Merge data set
def createDataset():
    dfHubspot = get_individual_page()
    dfWeight = retrieveZWEMData('1', True)

    weight = filteredWeight(dfWeight)

    # Merge dataframes using inner join on first and last name
    mergedDF = pd.merge(weight, dfHubspot, on=['first_name', 'last_name'], how='inner') 
    mergedDF = mergedDF.dropna(subset=['num_visits'])

    # Retrieve waist, glucose and HbA1c
    for i in mtypeid:
        df = retrieveZWEMData(i[0], False)
        filtered_df = firstAndLastDate(df, False, i[1])
        mergedDF = pd.merge(mergedDF, filtered_df, on=['first_name', 'last_name'], how='inner')

    # Drop some redundant columns
    finalDF = mergedDF.drop(['personid', 'first_weight', 'last_weight',  'first_weight_date', 'first_waist_date', 'first_glucose_date', 'first_HbA1c_date', 'last_weight_date', 'last_waist_date', 'last_glucose_date', 'last_HbA1c_date', 'birthdate'], axis=1)

    # Change waist to wthr and add first age
    featureEngineering(finalDF)

    # Filter zero's and isnull values
    finalDF = finalDF[(finalDF['HbA1c_duration'] != 0) & (~finalDF['HbA1c_duration'].isnull()) & (finalDF['glucose_duration'] != 0) & (~finalDF['glucose_duration'].isnull()) & (finalDF['waist_duration'] != 0) & (~finalDF['waist_duration'].isnull())]

    # Drop more redundant columns
    finalDF = finalDF.drop(['weight_duration', 'waist_duration', 'glucose_duration', 'HbA1c_duration', 'length'], axis=1)

    completeDF = splitFirstandLastMeasurement(finalDF)
    completeDF['classification'] = completeDF.apply(diabetes, axis=1)
    completeDF = completeDF.drop(['waist', 'first_name', 'last_name', 'gender'], axis=1)

    return completeDF

def machineLearningAlgorithm(df, models, number):
    # Split dependent and independent variables
    X = df.drop('classification', axis=1)
    y = df['classification'] 

    # Split data into test and train data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # k-fold cross validation
    num_folds = 10
    kfold = KFold(n_splits=num_folds, random_state=42, shuffle=True)

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
 
    # Create model
    model = algorithm(models[number])

    # k scores on scaled training data
    scores = cross_val_score(model, X_train_scaled, y_train, cv=kfold)
    print(scores, scores.mean())

    # Train model on entire training set
    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = model.predict(X_test_scaled)

    # Classification report
    target_names = ['without diabetes', 'with diabetes']
    print(metrics.classification_report(y_test, y_pred, target_names=target_names))

    # ROC curve
    roc(model, X_test_scaled, y_test)

    # Confusion matrix
    matrix = metrics.confusion_matrix(y_test, y_pred)
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Determine importance of variables
    explainer = shap.KernelExplainer(model.predict_proba, X_train_scaled)
    shap_values = explainer.shap_values(X_test_scaled)
    shap.summary_plot(shap_values, X_test) 

def algorithm(algorithm):
    if algorithm == 'SVM':                                              # Support Vector Machine
        svm = SVC(kernel='linear', probability=True) 
        return svm
    elif algorithm == 'LR':                                             # Logistic Regression
        lr = LogisticRegression(random_state=16)
        return lr
    elif algorithm == 'KNN':                                            # K-Nearest Neighbors
        knn = KNeighborsClassifier(n_neighbors=14)
        return knn
    elif algorithm == 'RF':                                             # Random Forest
        rf = RandomForestClassifier()
        return rf
    elif algorithm == 'NB':                                             # Naive Bayes
        nb = GaussianNB()
        return nb
    else:                                                               # Decision Tree
        dt = DecisionTreeClassifier(criterion="entropy", max_depth=7)
        return dt
    
def roc(model, X_test_scaled, y_test):
    y_pred_proba = model.predict_proba(X_test_scaled)[::,1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label="data 1, auc="+str(auc))
    plt.legend(loc=4)
    plt.show()

def confusionmatrix(cnf_matrix):
    class_names=[0,1] # name  of classes
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    # create heatmap
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

def findBestK(y):
    k_values = [i for i in range (1,31)]
    scores = []
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k)
        score = cross_val_score(model, X, y, cv=5)
        scores.append(np.mean(score))

    sns.lineplot(x = k_values, y = scores, marker = 'o')
    plt.xlabel("K Values")
    plt.ylabel("Accuracy Score")
    plt.show()

def optimalDepth(df):
    # Split dependent and independent variables
    X = df.drop('classification', axis=1)
    y = df['classification'] 

    max_depth_values = np.arange(1, 21)

    # Store the cross-validation scores for each max depth
    cv_scores = []

    for depth in max_depth_values:
        dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
        scores = cross_val_score(dt, X, y, cv=10, scoring='accuracy')  # Adjust scoring based on your problem
        cv_scores.append(np.mean(scores))
    
    plt.plot(max_depth_values, cv_scores, marker='o')
    plt.xlabel('Max Depth')
    plt.ylabel('Cross-Validation Accuracy')
    plt.title('Decision Tree Hyperparameter Tuning')
    plt.grid(True)
    plt.show()

    # Identify the optimal max depth
    optimal_max_depth = max_depth_values[np.argmax(cv_scores)]
    print("Optimal Max Depth: ", optimal_max_depth)


def optimalRF(df):
    X = df.drop('classification', axis=1)
    y = df['classification'] 

    # Split data into test and train data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_classifier = RandomForestClassifier(random_state=42)

    # Define the hyperparameter grid to search
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [7, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    # Create a GridSearchCV object
    grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, scoring='accuracy')

    # Fit the GridSearchCV object to the data
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters from the grid search
    best_params = grid_search.best_params_
    print('Best Hyperparameters: ', best_params)

    # Use the best model to make predictions on the test set
    best_rf_model = grid_search.best_estimator_
    y_pred = best_rf_model.predict(X_test)

    target_names = ['without diabetes', 'with diabetes']
    print(metrics.classification_report(y_test, y_pred, target_names=target_names))


# <----------------- MAIN -----------------> #
models = ['SVM', 'LR', 'KNN', 'RF', 'NB', 'DT']
df = createDataset()
# Change last parameter to switch between the machine learning algorithms
machineLearningAlgorithm(df, models, 0)
