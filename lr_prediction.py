#functions
import mysql.connector
import pandas as pd
from datetime import datetime, timedelta
import numpy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


#function to get the next 20 days for prediction
def get_next_20days(last_row):
    # Specify the starting date
    start_date = last_row

    # Create a list to store the next 7 days
    next_20_days = []

    # Add the starting date to the list
    next_20_days.append(start_date + timedelta(days=1))

    # Iterate through the next 6 days
    for i in range(19):
        # Add one day to the previous date and append it to the list
        next_20_days.append(next_20_days[-1] + timedelta(days=1))

    return next_20_days

#linear regression prediction function
def predict(df,name):
    #create a model to predict official currency
    # Extract the day, month, and year from the 'date_column'
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    # Create a new data frame with the extracted values
    X = df[['day', 'month', 'year']]
    Y=df[['value_ars']]
    #get the last row
    last_row = df['date'].tail(1)
    l=get_next_20days(last_row)
    
    x_train, x_test, y_train, y_test = train_test_split(X,Y, train_size=0.7, test_size=0.3, random_state=1)
    model = LinearRegression()
    model.fit(x_train, y_train)
    days=[]
    new_p = pd.DataFrame(columns=['day', 'month', 'year', 'prediction', 'type'])
    for i in range(len(l)):
        d=l[i].dt.day
        m=l[i].dt.month
        y=l[i].dt.year
        #print(d.values)
        #print(type(d))
        days.append(d.values)
        
        p = pd.DataFrame({'day': d.values, 'month': m.values, 'year':y.values })
        pr=model.predict(p)
        print(type(pr.ravel()))
        
        new_p = new_p.append({'day': d.values[0], 'month': m.values[0], 'year':y.values[0], 'prediction': pr.ravel()[0], 'type':name }, ignore_index=True)

    return new_p

# Crea la conexión a la base de datos
cnx = mysql.connector.connect(
    host="arg-financial-market.cwtbtlx3yqpd.us-east-1.rds.amazonaws.com",  # El host de la base de datos
    user="root",  # El nombre de usuario
    password="gqLGyg85g2RGH8anXtmH",  # La contraseña del usuario
    database="arg_financial_market"  # El nombre de la base de datos
)

# Crea el cursor
cursor = cnx.cursor()


print("\n ------------------------------------------")
print("Reading data from AWS database")
print("------------------------------------------")


# Ejecuta la consulta
cursor.execute("""SELECT i.instrument_name,tr.instrument,t.day,t.month,t.year,tr.volume, tr.max_price_ars, tr.min_price_ars, tr.value_ars, tr.open_price_ars 
               FROM arg_financial_market.time t JOIN  arg_financial_market.transaction tr ON t.id=tr.time JOIN  arg_financial_market.instrument i ON tr.instrument=i.id  """)
results = cursor.fetchall()
print(results)
# Crea el DataFrame a partir de los resultados
data = pd.DataFrame(results,columns=['instrument','subinstrument', 'day', 'month', 'year', 'volume', 'max_price_ars', 'min_price_ars', 'value_ars', 'open_price_ars'])
data['date'] = pd.to_datetime(data[['day','month','year']], format={'date': '%d-%m-%Y'})

df = data.copy()

# Create dataframe for each type

#create a dataframe for each of the instrument
df_official = df[df['instrument'].str.match('Official')]
df_currency = df[df['instrument'].str.match('Currency')]
df_financial_market = df[df['instrument'].str.match('Financial market')]
df_crypto = df[df['instrument'].str.match('Cripto Currency')]
#create a data frame with the mean for currency and financial market
df_mean_currency = pd.DataFrame(df_currency.groupby(['date'])['value_ars'].mean())
df_mean_currency['date']=df_mean_currency.index
df_mean_financial_market = pd.DataFrame(df_financial_market.groupby(['date'])['value_ars'].mean())
df_mean_financial_market['date']=df_mean_financial_market.index



print("\n ------------------------------------------")
print("Linear Regression Prediction starts")
print("------------------------------------------")

official_pr=predict(df_official,'Official')
currency_pr=predict(df_currency,'Currency')
financial_market_pr=predict(df_financial_market,'Financial Market')
crypto_pr=predict(df_crypto,'Cripto')


query = "DELETE FROM arg_financial_market.predictions_lr"
cursor.execute(query)

print("\n ------------------------------------------")
print("Linear regression prediction ends")
print("------------------------------------------")

sets = [official_pr,currency_pr,financial_market_pr,crypto_pr]

print("\n ------------------------------------------")
print("Inserting predicted values into predictions table")
print("------------------------------------------")

for set in sets:
    for _, row in set.iterrows():
        tup_row = tuple(row)
        query = f"INSERT INTO arg_financial_market.predictions_lr (day, month, year, prediction, type) VALUES ({tup_row[0]}, {tup_row[1]}, {tup_row[2]}, {tup_row[3]}, '{tup_row[4]}');"
        cursor.execute(query)
    
print("\n ------------------------------------------")
print("Predicted values inserted")
print("------------------------------------------")


# Update changes
cnx.commit()

print("\n ------------------------------------------")
print("Changes commited, closing conection.")
print("------------------------------------------")

# Close conection
cnx.close()