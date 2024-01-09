import numpy as np
import requests
from credentials import API_key, conn_string
import pandas as pd
import pypyodbc as odbc
import matplotlib.pyplot as plt

ALL_ENDPOINT = 'https://api.hubapi.com/contacts/v1/lists/4/contacts/all?count=300' # set count to the number of members in the list
PROFILE_ENDPOINT = 'https://api.hubapi.com/contacts/v1/contact/vid/'

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

# Extract ZWEM data
def extractZWEM(mti, convertFlag):
    conn = odbc.connect(conn_string)

    sql = '''
    SELECT M.personid, M.mtypeid, M.measurementdate, M.measuredvalue, P.voornaam AS first_name, P.achternaam AS last_name, P.lengte AS length
    FROM dbo.Measurements AS M
    INNER JOIN dbo.Personen AS P
    ON M.personid=P.id
    WHERE M.mtypeid=?
    ORDER BY M.measurementdate
    '''

    # Execute SQL statement and store in DataFrame
    cursor = conn.cursor()
    cursor.execute(sql, (mti,))

    dataset = cursor.fetchall()
    columns = [column[0] for column in cursor.description] 
    
    df = pd.DataFrame(dataset, columns=columns)

    # Change 'Object' type to 'datetime64[ns]' type
    df['measurementdate'] = pd.to_datetime(df['measurementdate'])

    first_measurement_idx = df.groupby('personid')['measurementdate'].idxmin()
    last_measurement_idx = df.groupby('personid')['measurementdate'].idxmax()

    first_last_measurements = df.loc[first_measurement_idx, ['first_name', 'last_name', 'personid', 'measurementdate', 'measuredvalue', 'length']]
    first_last_measurements['last_measurementdate'] = df.loc[last_measurement_idx, 'measurementdate'].values
    first_last_measurements['last_value'] = df.loc[last_measurement_idx, 'measuredvalue'].values
    first_last_measurements.rename(columns={'measurementdate': 'first_measurementdate', 'measuredvalue': 'first_value'}, inplace=True)
    first_last_measurements['duration'] = (first_last_measurements['last_measurementdate'] - first_last_measurements['first_measurementdate']).astype(str).str.extract('(\d+)').astype(int)

    if convertFlag:
        if mti == 1:
            # Change Weight to BMI
            first_last_measurements['last_BMI'] = first_last_measurements['last_value'] / ((first_last_measurements['length']/100)*(first_last_measurements['length']/100)) 
            first_last_measurements['first_BMI'] = first_last_measurements['first_value'] / ((first_last_measurements['length']/100)*(first_last_measurements['length']/100))
            first_last_measurements['difference'] = (first_last_measurements['last_BMI'] - first_last_measurements['first_BMI'])
            first_last_measurements = first_last_measurements.drop(['last_BMI', 'first_BMI'], axis=1)
        elif mti == 3:
            # Change waist to WtHr
            first_last_measurements['last_WtHr'] = first_last_measurements['last_value'] / first_last_measurements['length']
            first_last_measurements['first_WtHr'] = first_last_measurements['first_value'] / first_last_measurements['length']
            first_last_measurements['difference'] = (first_last_measurements['last_WtHr'] - first_last_measurements['first_WtHr'])
            first_last_measurements = first_last_measurements.drop(['last_WtHr', 'first_WtHr'], axis=1)
            print(first_last_measurements)
        else:
            first_last_measurements['difference'] = (first_last_measurements['last_value'] - first_last_measurements['first_value'])
    else:
        first_last_measurements['difference'] = (first_last_measurements['last_value'] - first_last_measurements['first_value'])
    
    finalDF = first_last_measurements.drop(['personid', 'first_measurementdate', 'last_measurementdate', 'first_value', 'last_value'], axis=1)
    return finalDF

# Combine ZWEM and Hubspot data, display online activity in bar chart
def mergeData(variable, convertFlag):
    df_zwem = extractZWEM(variable[0], convertFlag)
    df_hubspot = get_individual_page()
    df_merged = pd.merge(df_zwem, df_hubspot, on=['first_name', 'last_name'], how='inner')
        
    num_bins = 11
    bins = np.linspace(variable[1][0], variable[1][1], num_bins, endpoint=True)

    df_merged['value_bin'] = pd.cut(df_merged['difference'], bins, right=False)
    df_merged['num_visits'] = pd.to_numeric(df_merged['num_visits'], errors='coerce')
    df_merged['num_views'] = pd.to_numeric(df_merged['num_views'], errors='coerce')
    df_grouped = df_merged.groupby('value_bin').agg({'num_visits': 'mean', 'num_views': 'mean', 'difference': 'count'}).reset_index()
    
    print(df_grouped)

    fig, ax = plt.subplots(figsize=(10, 6))

    bar_width = 0.4
    opacity = 0.8

    ax.bar(df_grouped.index, df_grouped['num_visits'], bar_width, alpha=opacity, color='b', label='Average Num-visits')
    ax.bar(df_grouped.index + bar_width, df_grouped['num_views'], bar_width, alpha=opacity, color='r', label='Average Num-views')

    ax.set_xlabel(variable[2])
    ax.set_ylabel('Number of views/visits')
    ax.set_title(variable[3])
    ax.set_xticks(df_grouped.index + bar_width / 2)
    ax.set_xticklabels(df_grouped['value_bin'].astype(str), rotation=45)
    ax.legend()

    plt.tight_layout()
    plt.show()

# Labels of corresponding variables
variables = [
                (1, (-16, 4), 'Weight Difference in kg', 'Average Num-visits and Num-views in Relation to Weight Difference'), 
                (3, (-9,1), 'Waist Circumference Difference in cm', 'Average Num-visits and Num-views in Relation to Waist Circumference Difference'), 
                (4,(-3, 2), 'Fasting Glucose Difference in mmol/L', 'Average Num-visits and Num-views in Relation to Fasting Glucose Difference'), 
                (5, (-8,1), 'HbA1c Difference in mmol/mol', 'Average Num-visits and Num-views in Relation to HbA1c Difference'), 
                (1, (-5.5,1.5), 'BMI Difference', 'Average Num-visits and Num-views in Relation to BMI Difference'), 
                (3, (-0.08,0.02), 'WtHr Difference in cm', 'Average Num-visits and Num-views in Relation to WtHr Difference')
            ]


# (variables[0], False) = weight, (variables[1], False) = waist, (variables[2], False) = glucose
# (variables[3], False) = HbA1c, (variables[4], True) = BMI and (variables[5], True) = WtHr
mergeData(variables[0], False)
