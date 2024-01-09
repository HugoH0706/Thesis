from matplotlib.dates import relativedelta
import pypyodbc as odbc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from credentials import conn_string

# SQL statement for BMI and WtHr
sql2 = '''
    SELECT M.personid, M.mtypeid, M.measurementdate, M.measuredvalue, P.lengte
    FROM dbo.Measurements AS M
    INNER JOIN dbo.Personen AS P
    ON M.personid=P.id
    WHERE M.mtypeid=?
    ORDER BY M.measurementdate
    '''
        
# SQL statement for weight, waist circumference, HbA1c and glucose        
sql = '''
    SELECT M.personid, M.mtypeid, M.measurementdate, M.measuredvalue
    FROM dbo.Measurements AS M
    WHERE M.mtypeid=?
    ORDER BY M.measurementdate
    '''

def plotGraph(overall_avg, label):
    plt.plot(overall_avg['new_date'], overall_avg['mean_value_difference'], linestyle='-', color='black')
    plt.xlabel('Month Offset (Months Since First Measurement)')
    plt.ylabel(label[0])
    plt.title(label[1])
    plt.grid(True)
    plt.show()

def waistToLengthRatioConverter(row):
    return (row['measuredvalue'] / row['lengte'])

def BMIConverter(row):
    return (row['measuredvalue'] / ((row['lengte']/100)*(row['lengte']/100)))

def individualGraph(conn, label, first_name, last_name):
    mti = label[2]

    cursor = conn.cursor()

    SQLIndividual = '''
    SELECT M.personid, M.mtypeid, M.measurementdate, M.measuredvalue, P.voornaam, P.achternaam, P.lengte
    FROM dbo.Measurements AS M
    INNER JOIN dbo.Personen AS P
    ON M.personid=P.id
    WHERE M.mtypeid=? AND P.voornaam=? AND P.achternaam=?
    ORDER BY M.measurementdate
    '''
    params = (mti, first_name, last_name)
    cursor.execute(SQLIndividual, params)

    dataset = cursor.fetchall()
    if not dataset:
        print("Person is not a member of the data set")
    else: 
        columns = [column[0] for column in cursor.description] 
        
        df = pd.DataFrame(dataset, columns=columns)
        
        df['measurementdate'] = pd.to_datetime(df['measurementdate'])

        df = df.sort_values(by='measurementdate')
        print(df)

        df['measurement_difference'] = df['measuredvalue'].diff()
        df['cumulative_difference'] = df['measurement_difference'].cumsum()
        df['cumulative_average_difference'] = df.groupby(df['measurementdate'].dt.to_period("M"))['cumulative_difference'].transform('mean')

        plt.figure(figsize=(10, 6))
        df.plot(x='measurementdate', y='cumulative_average_difference', kind='line', marker='o', color='skyblue', label=None)
        plt.title(label[1])
        plt.xlabel('Measurement Date')
        plt.ylabel(label[0])
        plt.xticks(rotation=45)
        plt.legend().set_visible(False)
        plt.show()


def graph(conn, label, convertFlag):
    mti = label[2]
    
    # Execute SQL statement and store in DataFrame
    cursor = conn.cursor()
    if convertFlag:
        cursor.execute(sql2, (mti,))
    else:
        cursor.execute(sql, (mti,))

    dataset = cursor.fetchall()
    columns = [column[0] for column in cursor.description] 
    
    df = pd.DataFrame(dataset, columns=columns)

    if convertFlag and mti == 1:
        df['measuredvalue'] = df.apply(BMIConverter, axis=1) #BMI
    if convertFlag and mti == 3:
        df['measuredvalue'] = df.apply(waistToLengthRatioConverter, axis=1) #WtHr

    # Change 'Object' type to 'datetime64[ns]' type
    df['measurementdate'] = pd.to_datetime(df['measurementdate'])

    # Add column month and year to DataFrame
    df['month'] = df['measurementdate'].dt.month
    df['year'] = df['measurementdate'].dt.year

    # Take the date of first measurement
    first_measured_date = df.groupby('personid')['measurementdate'].min().reset_index()
    first_measured_date.rename(columns={'measurementdate': 'first_measured_date'}, inplace=True)

    # Merge first_measured_date DataFrame into main DataFrame
    df = df.merge(first_measured_date, on='personid')

    # Calculate month offset
    df['month_offset'] = ((df['measurementdate'].dt.year - df['first_measured_date'].dt.year) * 12) + (df['measurementdate'].dt.month - df['first_measured_date'].dt.month)+1
    df['month_offset'] = np.where(df['measurementdate'] == df['first_measured_date'], 0, df['month_offset'])

    df['value_difference'] = (df.groupby('personid')['measuredvalue'].transform('first') - df['measuredvalue'])*-1 

    # Calculated value difference, amount of people and amount of measurements per month
    overall_avg = df.groupby('month_offset').agg(
        amount_of_people=('personid', 'nunique'),
        mean_value_difference=('value_difference', 'mean'),
        total_measurements=('value_difference', 'size')
    ).reset_index()

    overall_avg = overall_avg.head(30)
    measurements = overall_avg['total_measurements'].sum()
    print("Total measurements: ", measurements)
    average = overall_avg['amount_of_people'].sum() / 30
    print("Average people: ", average)

    plotGraph(overall_avg, label)
    cursor.close()

def createGraph2019(label):
    conn = odbc.connect(conn_string)
    mti = label[2]
    
    cursor = conn.cursor()
    cursor.execute(sql, (mti,))

    dataset = cursor.fetchall()
    columns = [column[0] for column in cursor.description] 
    
    df = pd.DataFrame(dataset, columns=columns)

    df['measurementdate'] = pd.to_datetime(df['measurementdate'])
    df = df.loc[df['measurementdate'] >= '2019-01-01']
    df['month'] = df['measurementdate'].dt.month
    df['year'] = df['measurementdate'].dt.year

    first_measured_date = df.groupby('personid')['measurementdate'].min().reset_index()
    first_measured_date.rename(columns={'measurementdate': 'first_measured_date'}, inplace=True)

    df = df.merge(first_measured_date, on='personid')
    df['month_offset'] = ((df['measurementdate'].dt.year - df['first_measured_date'].dt.year) * 12) + (df['measurementdate'].dt.month - df['first_measured_date'].dt.month)+1
    df['month_offset'] = np.where(df['measurementdate'] == df['first_measured_date'], 0, df['month_offset'])
    df['value_difference'] = (df.groupby('personid')['measuredvalue'].transform('first') - df['measuredvalue'])*-1 

    overall_avg = df.groupby('month_offset').agg(
        amount_of_people=('personid', 'nunique'),
        mean_value_difference=('value_difference', 'mean'),
        total_measurements=('value_difference', 'size')
    ).reset_index()

    cursor.close()

    overall_avg['new_date'] = pd.to_datetime('2019-01-01') + overall_avg['month_offset'].apply(lambda x: relativedelta(months=x))
    plotGraph(overall_avg, label)

    return overall_avg


labels = [
    ('Weight (kg)','Average Body Weight Difference Over The Course Of 30 Months', 1),
    ('Waist Circumference (cm)','Average Waist Circumference Difference Over The Course Of 30 Months', 3),
    ('Glucose (mmol/l)','Average Glucose Difference Over The Course Of 30 Months', 4),
    ('HbA1c (mmol/mol)','Average HbA1c Difference Over The Course Of 30 Months', 5),
    ('BMI','Average BMI Difference Over The Course Of 30 Months', 1),
    ('Waist to Height Ratio cm','Average WtHR Difference Over The Course Of 30 Months', 3)
]

labels2 = [
    ('Weight (kg)','Average Body Weight Difference', 1),
    ('Waist Circumference (cm)','Average Waist Circumference Difference', 3),
    ('Glucose (mmol/l)','Average Glucose Difference', 4),
    ('HbA1c (mmol/mol)','Average HbA1c Difference', 5)
]

conn = odbc.connect(conn_string)

# graph(conn, labels[0], False) = Weight, graph(conn, labels[1], False) = Waist Circumference, graph(conn, labels[2], False) = Glucose
# graph(conn, labels[3], False) = HbA1c, graph(conn, labels[4], True) = BMI, graph(conn, labels[5], True) = WtHr 
#graph(conn, labels[0], False)

createGraph2019(labels[0])