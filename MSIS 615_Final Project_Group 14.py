# IMPORTING VARIOUS LIBRARIES AND MODULES
import re
import time
import sqlite3
import math
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
import random
import time
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import researchpy as rp
import statistics

# FETCHING THE DATA FROM THE WEBSITE
s=Service("C:/Users/Administrator/Documents/Business Programming/Jupyter/BP - Group Project/chromedriver.exe")
driver=webdriver.Chrome(service=s)
URL_pattern_str="https://www.cargurus.com/Cars/spt_used_cars?px8324=p2&cgcid=2161&cgagid=910472&sourceContext=cargurus&ax8324=167427153&type=GoogleAdWordsSearch&kw=cargurus&matchtype=p&ad=649338002818&placement=&networktype=g&device=c&devicemodel=&adposition=&physloc=9001987&intloc=&aceid=&cid=141954540&agid=7773125700&tgtid=kwd-11550557270&fid=&gclid=Cj0KCQjw3a2iBhCFARIsAD4jQB2s1lSAWX5xPeB1nghwr_VtRX29T_lBiEXKnwcRUGnWYvlKi2Ean9QaAjDpEALw_wcB#resultsPage=$NUM$"

emi = []
miles = []
name = []
price = []
location = []
year = []
brand = []
make = []
city = []
state = []
ybm = []

m = 0
while m < 60:
    ran = random.randint(1,10)
    time.sleep(ran)
    page_URL=URL_pattern_str.replace("$NUM$",str(1))
    driver.get(page_URL)
    page_content = driver.page_source
    
    matches_emi=re.compile(r"<span>\$([0-9,]+)\/mo est\.\*<\/span>",re.S|re.I).findall(page_content)
    for i in matches_emi:
        emi.append(i)

    matches_miles=re.compile(r"<span>(\d{1,3},\d{3})? mi<\/span>",re.S|re.I).findall(page_content)
    for i in matches_miles:
        miles.append(i)

    matches_ybm = re.compile(r"<h4 class=\"vO42pn\" title=\"\d{0}([0-9,]+)\s(.*?)\s(.*?)\">.*?<\/h4>", re.S|re.I).findall(page_content)
    for i in matches_ybm:
        ybm.append(i)

    matches_price = re.compile(r"<span class=\"JzvPHo\" data-cg-ft=\"srp-listing-blade-price-and-payment\">(\W\d{1,3},\d{3})", re.S|re.I).findall(page_content)
    for i in matches_price:
        price.append(i)

    matches_location = re.compile(r"<span>(\w+),\s(.*?)\s\W\d{1,3}\s.*?\W<\/span>", re.S|re.I).findall(page_content)
    for i in matches_location:
        location.append(i)

    for i in matches_ybm:
        year.append(i[0])
        brand.append(i[1])
        make.append(i[2])

    for i in matches_location:
        city.append(i[0])
        state.append(i[1])
        
    m = m + 1

# LOADING ALL THE DATA INTO SQLITE DATABASE
conn = sqlite3.connect('CarguruFinal.db')
c = conn.cursor()
c.execute("DROP TABLE IF EXISTS Cars")
c.execute("CREATE TABLE Cars(\
              Brand varchar(100), \
              Model varchar(100), \
              Year int(4), \
              MilesDriven int(10), \
              Price int(10), \
              EMI int(10), \
              City varchar(10), \
              State char(2))")

for i in range(len(city)):
    values = (brand[i], make[i], year[i], miles[i], price[i], emi[i], city[i], state[i])
    query = "INSERT INTO Cars VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
    c.execute(query, values)

conn.commit()

c.execute("Select * FROM Cars")
data = c.fetchall()

# Print the data to the console
for row in data:
    print(row)

# DATA CLEANING    
# REMOVING '$' AND ',' BY USING UPDATE QUERY
conn.commit()
c.execute("UPDATE [Cars] SET[EMI]=REPLACE([EMI], ',', '')")
data = c.fetchall()

conn.commit()
c.execute("UPDATE [Cars] SET[EMI]=REPLACE([EMI], '$', '')")
data = c.fetchall()

conn.commit()
c.execute("UPDATE [Cars] SET[MilesDriven]=REPLACE([MilesDriven], ',', '')")
data = c.fetchall()

conn.commit()
c.execute("UPDATE [Cars] SET[Price]=REPLACE([Price], ',', '')")
data = c.fetchall()

conn.commit()
c.execute("UPDATE [Cars] SET[Price]=REPLACE([Price], '$', '')")
data = c.fetchall()

conn.commit()
c.execute("Select * From Cars")
data = c.fetchall()


# Print the data to the console
for row in data:
    print(row)

# DATA ANALYTICS
# Regression 1
# Connect to the database
conn = sqlite3.connect('CarguruFinal.db')

# Retrieve the data from the database
df = pd.read_sql_query("SELECT * FROM Cars", conn)

# Prepare the data for regression1
X = df['Price'].values.reshape(-1,1)
y = df['MilesDriven'].values.reshape(-1,1)

slope, intercept, r_value, p_value, std_err = stats.linregress(X[:,0], y[:,0])
# Calculate R-squared
r_squared = r_value**2

print('Slope:', slope)
print('Intercept:', intercept)
print('Std Err:', std_err)
print('R-squared:', r_squared)
print('p-value:', p_value)

# Regression 2
# Retrieve the data from the database
df2 = pd.read_sql_query("SELECT * FROM Cars", conn)

# Prepare the data for regression2
X = df2[['Price']]
y = df2[['MilesDriven','EMI']]

reg = LinearRegression().fit(X, y)
coefficients = reg.coef_
intercept = reg.intercept_
r_squared = reg.score(X, y)

print('Slope:', slope)
print('Intercept:', intercept)
print('Std Err:', std_err)
print('R-squared:', r_squared)
print('p-value:', p_value)

# Retrieve the data from the database
dfl = pd.read_sql_query("SELECT * FROM Cars", conn)
print(dfl)

#Performing Linear regression 
mlr = smf.ols(formula="Price~MilesDriven",data=dfl).fit()
print(mlr.summary())

# Retrieve the data from the database
df = pd.read_sql_query("SELECT * FROM Cars", conn)

# Scatter plot of Price vs MilesDriven
plt.scatter(df['MilesDriven'], df['Price'])

# Linear regression line plot
plt.plot(df['MilesDriven'], mlr.predict(df['MilesDriven']), color='red')

# Set the labels and title
plt.xlabel('MilesDriven')
plt.ylabel('Price')
plt.title('Regression')

# Show the plot
plt.show()

# Connect to the database
conn = sqlite3.connect('CarguruFinal.db')

# Multiple Linear Regression
# Retrieve the data from the database
dfm = pd.read_sql_query("SELECT * FROM Cars", conn)
print(dfm)

#Performing multiple regression 
mlrm = smf.ols(formula="Price~EMI+MilesDriven",data=dfm).fit()
print(mlrm.summary())

# Scatter plot of Price vs MilesDriven+EMI
plt.scatter(df['MilesDriven'], df['Price'], color='blue', label='MilesDriven')
plt.scatter(df['EMI'], df['Price'], color='red', label='EMI')

# predicting values based on the regression equation
predicted = mlr.predict(df[['EMI', 'MilesDriven']])

# Plot the predicted values
plt.plot(df['MilesDriven'], predicted, color='green', label='Regression Line')

# Set the labels and legend
plt.xlabel('Miles Driven / EMI')
plt.ylabel('Price')
plt.legend()
plt.title('Multiple Regression')

# Plotting multiple regression
plt.show()

# DATA VISUALIZATION
# Loading the dataset from the "CARS" table
query = "SELECT * FROM CARS"
dfa = pd.read_sql(query, conn)

# Plotting Bar Graph of Price vs State
state_price_df = dfa[['State', 'Price']]
state_price_grouped = state_price_df.groupby(['State']).mean()
plt.bar(state_price_grouped.index, state_price_grouped['Price'])
plt.xticks(rotation=90)
plt.xlabel('State')
plt.ylabel('Average Price')
plt.title('Average Car Price by State')
plt.show()

# Loading the dataset from the "CARS" table
query = "SELECT * FROM CARS"
df = pd.read_sql(query, conn)
#Plot the bar graph for miles driven and state
sns.barplot(df['State'],df['MilesDriven'])

#Loading the table only for state MA
query = "SELECT * FROM CARS Where State='MA'"
dfp = pd.read_sql(query, conn)
#Using describe method for summary
dfp['Price'].describe()
print(rp.summary_cont(dfp['Price']))

# Load the dataset from the "CARS" table
query = "SELECT State, Price FROM CARS"
dfs = pd.read_sql(query, conn)

# Calculating the standard deviation of the "Price" variable for each state using the "groupby" function
dfs['std'] = dfs.groupby('State')['Price'].transform(lambda x: statistics.stdev(x))

# Print the summary statistics
print(dfs.groupby('State').agg({'Price': ['count', 'mean', 'min', 'max', 'std']}))


