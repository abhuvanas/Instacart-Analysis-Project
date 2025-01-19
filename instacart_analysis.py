import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

orders = pd.read_csv('archive/orders.csv')
aisles = pd.read_csv('archive/aisles.csv')
departments = pd.read_csv('archive/departments.csv')
products = pd.read_csv('archive/products.csv')
order_products_prior = pd.read_csv('archive/order_products__prior.csv')
order_products_train = pd.read_csv('archive/order_products__train.csv')

df = pd.read_csv('archive/orders.csv')

print(orders.head())

#Data Summary 
print(orders.info())
print(orders.describe())

#Merge data for DA
merged_data = pd.merge(order_products_prior, products, on='product_id', how='left')
merged_data = pd.merge(merged_data, orders, on='order_id', how='left')
print(merged_data.head())
#Feature engineering
merged_data['total_user_orders'] = merged_data.groupby('user_id')['order_number'].transform('max')
merged_data['product_popularity'] = merged_data.groupby('product_id')['product_id'].transform('count')
merged_data['user_product_orders'] = merged_data.groupby(['user_id', 'product_id'])['product_id'].transform('count')
merged_data['avg_days_between_orders'] = merged_data.groupby('user_id')['days_since_prior_order'].transform('mean')
#buidling a model


#Feature building
sample_data = merged_data.sample(frac=0.1, random_state=42)  # Take 10% of the data
X = merged_data[['order_number', 'add_to_cart_order','total_user_orders', 'product_popularity', 'days_since_prior_order', 'user_product_orders', 'avg_days_between_orders']]
y = merged_data['reordered']

# deal with missing valuees
X = X.fillna(0)
y = y.fillna(0)



#Split data 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Model
model = LogisticRegression()
model.fit(X_train, y_train)

#predict and evluate
y_pred = model.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))


#figuring out the most popular products
popular_products = order_products_prior['product_id'].value_counts().head(10)
popular_products_names = products[products['product_id'].isin(popular_products.index)]
print(popular_products_names)
plt.figure(figsize=(12, 6))
sns.barplot(x=popular_products_names['product_name'], y=popular_products.values, palette='plasma')
plt.title('Top 10 Most Popular Products')
plt.xlabel('Product Name')
plt.ylabel('Number of Orders')
plt.xticks(rotation=45)
plt.show(block = False)
plt.show()

#most popular product is 24852

order_heatmap = orders.groupby(['order_dow', 'order_hour_of_day']).size ().unstack()
plt.figure(figsize=(12, 8))
sns.heatmap(order_heatmap, cmap='coolwarm', annot=False)
plt.title('Heatmap of Orders by Day and Hour')
plt.xlabel('Hour of Day')
plt.ylabel('Day of Week')
plt.show(block=False)
plt.show()