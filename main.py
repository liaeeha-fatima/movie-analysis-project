import pandas as pd

import pandas as pd

customers = pd.read_csv(r"C:\Users\Sahand System\python\churn\customers.csv")
orders = pd.read_csv(r"C:\Users\Sahand System\python\churn\orders.csv")

print(customers.head())
print(orders.head())

print(customers.info())
print(orders.info())

print(customers.describe())
print(orders.describe())

#  Missing Values
print(customers.isnull().sum())
print(orders.isnull().sum())

# Duplicate
print("Duplicate Customer:" , customers.duplicated().sum())
print("Duplicate Orders", orders.duplicated().sum())

# Standardizing text columns
  #Gender
customers['gender']=customers['gender'].replace({'m':'M','f':'F'}).fillna('Other')
  #Country
customers['country']= customers['country'].fillna('DE')
  #aquisition channel  
customers['acquisition_channel'] = customers['acquisition_channel'].fillna('Organic')
  #payment Method
orders['payment_method'] = orders['payment_method'].fillna('Other')  

   #age
customers = customers[(customers['age']>=18) & (customers['age'] <=65)]
   #revnue >0 
orders = orders[orders['revenue'] >0] 


#data Type conversion
customers['signup_date'] = pd.to_datetime(customers['signup_date'])
orders['order_date'] = pd.to_datetime(orders['order_date'])
customers['customer_id'] = customers['customer_id'].astype(int)
orders['customer_id'] = orders['customer_id'].astype(int)
orders['order_id'] = orders['order_id'].astype(int)


#Reference Date
reference_date = orders['order_date'].max()
reference_date

#Last Order Date
last_order = (
    orders
    .groupby('customer_id')['order_date']
    .max()
    .reset_index()
    .rename(columns={'order_date':'last_order_date'})

)

#join 
customers_fe = customers.merge(
    last_order,
    on='customer_id',
    how='left'
)
# Days Since Last Order
customers_fe['days_since_last_order'] = (
    reference_date - customers_fe['last_order_date']
).dt.days

#churn
customers_fe['is_churn'] = customers_fe['days_since_last_order']>90

#order Count
order_count = (
    orders
    .groupby('customer_id')['order_id']
    .count()
    .reset_index(name='order_count')
)

#total_revenue
total_revenue = (
    orders
    .groupby('customer_id')['revenue']
    .sum()
    .reset_index(name='total_revenue')

)
#Join the final Features
customers_fe = (
    customers_fe
    .merge(order_count, on='customer_id', how='left')
    .merge(total_revenue, on='customer_id', how='left')
)
#Logical NaN replacement
customers_fe[['order_count','total_revenue']]=(
    customers_fe[['order_count','total_revenue']].fillna(0)
)
#Final Feature Table Quality Check
customers_fe.info()
customers_fe[['days_since_last_order','order_count','total_revenue']].describe()

#churn rate
churn_rate = customers_fe['is_churn'].mean()
print(f"churn Rate: {churn_rate:.2%}")

# Churned vs Active
grouped = customers_fe.groupby('is_churn')[['order_count', 'total_revenue']].mean()
print(grouped)

#correct churn
print (customers_fe['days_since_last_order'].describe())
print(customers_fe.columns)

CHURN_DAY =180
customers_fe['is_churn'] = (
    customers_fe['days_since_last_order'] > CHURN_DAY
)

print("new churn rate:", customers_fe['is_churn'].mean())


grouped = (
    customers_fe
    .groupby('is_churn')[['order_count', 'total_revenue']]
    .mean()
)
print(grouped)

print(customers_fe['days_since_last_order'].describe())
print(customers_fe['is_churn'].value_counts())
# باز هم churn بالاست و با این خطوط دارم اصلاح می کنم

churn_threshold = customers_fe['days_since_last_order'].quantile(0.75)

customers_fe['is_churn'] = (
    customers_fe['days_since_last_order']>churn_threshold
)

print(customers_fe['is_churn'].value_counts(normalize=True))


#T_test
from scipy.stats import ttest_ind
active = customers_fe[customers_fe['is_churn'] == False]['order_count']
churned = customers_fe[customers_fe['is_churn'] == True]['order_count']
t_stat, p_value = ttest_ind(active, churned, equal_var=False)
print("order count p_value", p_value)

#Total Revenue
active_rev = customers_fe[customers_fe['is_churn'] == False]['total_revenue']
churned_rev = customers_fe[customers_fe['is_churn'] == True]['total_revenue']
t_stat, p_value = ttest_ind(active_rev, churned_rev, equal_var=False)
print("Revenue p_value:", p_value)

#Acquisition Channel
from scipy.stats import chi2_contingency
contingency = pd.crosstab(
    customers_fe['acquisition_channel'],
    customers_fe['is_churn']
)
chi2, p, dof,excepted = chi2_contingency(contingency)
print("Acquisition Channel p_value:",p)

#Gender
contingency = pd.crosstab(
    customers_fe['gender'],
    customers_fe['is_churn']
)
chi2,p,dof,excepted = chi2_contingency(contingency)
print("Gender p_value:",p)

