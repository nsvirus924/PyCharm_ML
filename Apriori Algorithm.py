## UNSUPERVISED LEARNING ##

## Apriori Algorithm
import openpyxl
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
df = pd.read_excel("D:\Online_Retail.xlsx")
print(df.head())

## Data Cleaning
#to shorten the dataset we only look at the dataset with sales in france

basket = (df[df['Country'] == "France"].groupby(['InvoiceNo','Description'])['Quantity']
    .sum().unstack().reset_index().fillna(0).set_index('InvoiceNo'))
print(basket)

#we again encode it and check details
def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1
basket_sets = basket.applymap(encode_units)
basket_sets.drop('POSTAGE', inplace = True, axis=1)
print(basket_sets)

#We generate frquent itemsets that have support atleast 7%
frequent_itemset = apriori(basket_sets, min_support=0.07, use_colnames=True)
rules = association_rules(frequent_itemset,
                      metric= "lift" ,
                      min_threshold=0.8,
                      )
print(rules.head())

##what if we add another constraints on rules
print(rules[(rules['lift'] >=6) &
      (rules['confidence'] >=0.8)])
"""That is how we create association rules using apriori algorithm which helps a lot 
in marketing business"""