import base64
from io import BytesIO
from flask import Flask, render_template, request
import matplotlib
from matplotlib.dates import relativedelta
import pypyodbc as odbc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from credentials import conn_string, personalPassword
      
sql = '''
    SELECT M.personid, M.mtypeid, M.measurementdate, M.measuredvalue
    FROM dbo.Measurements AS M
    WHERE M.mtypeid=?
    ORDER BY M.measurementdate
    '''

labels = [
    ('Gewicht Verschil (kg)','Gemiddelde Gewicht Verschil 2019-heden', 1),
    ('Buikomvang Verschil (cm)','Gemiddelde Buikomvang Verschil 2019-heden', 3),
    ('Glucose Verschil (mmol/l)','Gemiddelde Glucose Verschil 2019-heden', 4),
    ('HbA1c Verschil (mmol/mol)','Gemiddelde HbA1c Verschil 2019-heden', 5),
]

labels2 = [
    ('Gewicht (kg)','Gemiddelde Gewicht Verschil', 1),
    ('Buikomvang (cm)','Gemiddelde Buikomvang Verschil', 3),
    ('Glucose (mmol/l)','Gemiddelde Glucose Verschil', 4),
    ('HbA1c (mmol/mol)','Gemiddelde HbA1c Verschil', 5)
]

app = Flask(__name__)

def createIndividualGraph(first_name, last_name, option, password):
    if password == personalPassword:
        if option == "Gewicht":
            mti = 1
            label = labels2[0]
        elif option == "Buikomvang":
            mti = 3
            label = labels2[1]
        elif option == "Glucose":
            mti = 4
            label = labels2[2]
        else:
            mti = 5
            label = labels2[3]
        
        conn = odbc.connect(conn_string)
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
            plt.text(0.5, 0.5, f"{first_name} {last_name} heeft geen { option } ZWEM metingen gedaan", ha='center', va='center', fontsize=12, color='red')
            plt.axis('off')
            plt.tight_layout()

            error_img_io = BytesIO()
            plt.savefig(error_img_io, format='png')
            error_img_io.seek(0)
            plt.close()

            error_img_base64 = base64.b64encode(error_img_io.getvalue()).decode('utf-8')
            return error_img_base64
        else:
            columns = [column[0] for column in cursor.description] 
            
            df = pd.DataFrame(dataset, columns=columns)
            df['measurementdate'] = pd.to_datetime(df['measurementdate'])
            df = df.sort_values(by='measurementdate')
            df['measurement_difference'] = df['measuredvalue'].diff()
            df['cumulative_difference'] = df['measurement_difference'].cumsum()
            df['cumulative_average_difference'] = df.groupby(df['measurementdate'].dt.to_period("M"))['cumulative_difference'].transform('mean')

            plt.figure(figsize=(10, 6))
            df.plot(x='measurementdate', y='cumulative_average_difference', kind='line', color='black', label=None)
            plt.title(label[1])
            plt.xlabel('Datum van Meting')
            plt.ylabel(label[0])
            plt.xticks(rotation=45)
            plt.legend().set_visible(False)

            img_io = BytesIO()
            plt.savefig(img_io, format='png')
            img_io.seek(0)
            plt.close()

            img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
            return img_base64
    else:
        print('nope')


def createImage(overall_avg, labels):
    plt.plot(overall_avg['date'], overall_avg['mean_value_difference'], linestyle='-', color='black')
    plt.xlabel('Maanden Sinds Eerste Meting')
    plt.ylabel(labels[0])
    plt.title(labels[1])
    
    img_io = BytesIO()
    plt.savefig(img_io, format='png')
    img_io.seek(0)
    plt.close()

    img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
    return img_base64

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

    overall_avg['date'] = pd.to_datetime('2019-01-01') + overall_avg['month_offset'].apply(lambda x: relativedelta(months=x))
    cursor.close()
    return overall_avg

@app.route('/', methods=['GET', 'POST'])
def home():
    # ---- First Plot ---- #
    overall_avg_1 = createGraph2019(labels[0])
    img_base64_1 = createImage(overall_avg_1, labels[0])
    
    # ---- Second Plot ---- #
    overall_avg_2 = createGraph2019(labels[1])
    img_base64_2 = createImage(overall_avg_2, labels[1])
    
    # ---- Third Plot ---- #
    overall_avg_3 = createGraph2019(labels[2])
    img_base64_3 = createImage(overall_avg_3, labels[2])

    # ---- Fourth Plot ---- #
    overall_avg_4 = createGraph2019(labels[3])
    img_base64_4 = createImage(overall_avg_4, labels[3])

    return render_template('index.html', img_data_1=img_base64_1, img_data_2=img_base64_2, img_data_3=img_base64_3, img_data_4=img_base64_4)


    
@app.route('/personal', methods=['GET', 'POST'])
def personal():
    options = ["Gewicht", "Buikomvang", "Glucose", "HbA1c"]
    if request.method == 'POST':
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        password = request.form['password']

        img_base64_individual1 = createIndividualGraph(first_name, last_name, options[0], password)
        img_base64_individual2 = createIndividualGraph(first_name, last_name, options[1], password)
        img_base64_individual3 = createIndividualGraph(first_name, last_name, options[2], password)
        img_base64_individual4 = createIndividualGraph(first_name, last_name, options[3], password)

        return render_template('personal.html', img_data_new_1=img_base64_individual1, img_data_new_2=img_base64_individual2, img_data_new_3=img_base64_individual3, 
                               img_data_new_4=img_base64_individual4, first_name=first_name, last_name=last_name)
    else:
        return render_template('personal.html', img_data_new_1=None, img_data_new_2=None, img_data_new_3=None, img_data_new_4=None, first_name=None, last_name=None)

if __name__ == '__main__':
    app.run(debug=True)