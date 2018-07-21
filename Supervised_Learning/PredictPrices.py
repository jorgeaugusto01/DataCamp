import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import datetime

dataset = pd.read_csv("petr4_1_2010_11_2017.csv", index_col="Date")

#dataset['Volume'] = dataset['Volume'].stack().str.replace(',','.').unstack()
dataset['Volume'] = dataset['Volume'].apply(lambda x: str(x.replace(',','.')))
dataset["Volume"] = dataset["Volume"].astype(float)
print(dataset['Volume'].head())


dataset['Variation'] = dataset['Close'].sub(dataset['Open'])
print(dataset.head())
df = dataset.ix['2010-01-01':'2017-04-31']
print(df.head())
print(df.describe())

ax = df['Close'].plot(title="Petr4", fontsize=10)
ax.set_xlabel("Years")
ax.set_ylabel("Prices")
#plt.show()

treino = df
x = treino.Open[:100]
y = treino.Close[:100]
plt.scatter(x,y,color='b')
plt.xlabel('preco de abertura')
plt.ylabel('preco de fechamento')
plt.axis([min(x),max(x),min(y),max(y)])
plt.autoscale('False')
#plt.show()

x = treino.High[:100]
y = treino.Close[:100]
plt.scatter(x,y,color='b')
plt.xlabel('preco da maxima')
plt.ylabel('preco de fechamento')
plt.axis([min(x),max(x),min(y),max(y)])
plt.autoscale('False')
#plt.show()

x = treino.Low[:100]
y = treino.Close[:100]
plt.scatter(x,y,color='b')
plt.xlabel('preco de Minima')
plt.ylabel('preco de fechamento')
plt.axis([min(x),max(x),min(y),max(y)])
plt.autoscale('False')
#plt.show()

x = treino.Volume[:100]
y = treino.Close[:100]
plt.scatter(x,y,color='b')
plt.xlabel('Volume')
plt.ylabel('preco de fechamento')
plt.axis([min(x),max(x),min(y),max(y)])
plt.autoscale('False')
plt.xticks(rotation=45)
plt.show()

features = ['Open','High','Low','Volume']
treino = treino[features]
y = df['Close']

X_treino, X_teste, y_treino, y_teste = train_test_split(treino, y,random_state=42)

lr_model = LinearRegression()

lr_model.fit(X_treino,y_treino)

print(lr_model.coef_)

print(lr_model.predict(X_teste)[:10])

print(y_teste[:10])

print(lr_model.predict(X_teste)[:10])

RMSE = mean_squared_error(y_teste, lr_model.predict(X_teste))**0.5

print(RMSE)

print(X_treino.head())

