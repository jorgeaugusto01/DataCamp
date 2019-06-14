#t-SNE visualization of grain dataset
#In the video, you saw t-SNE applied to the iris dataset. In this exercise, you'll apply t-SNE to the
# grain samples data and inspect the resulting t-SNE features using a scatter plot.
# You are given an array samples of grain samples and a list variety_numbers giving the variety number of each grain sample.z

from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

seeds = pd.read_csv('DataCamp\DataSets\seeds\\seeds.csv')
varietisSeeds = pd.read_csv('DataCamp\DataSets\seeds\\varietiesSeeds.csv')

varietisSeedsNumbers = pd.read_csv('DataCamp\DataSets\seeds\\varietiesSeedsNumbers.csv')
stockMovements = pd.read_csv('portfolio_equity_curve.csv').iloc[2:]
#stockMovements = pd.read_csv('precos_normalizados.csv').iloc[2:]
#stockMovements = pd.read_csv('sumario_15.csv').iloc[2:].set_index(['cod_neg']).join(pd.read_csv('sumario_30.csv').iloc[2:].set_index(['cod_neg']), rsuffix='30').join(pd.read_csv('sumario_60.csv').iloc[2:].set_index(['cod_neg']), rsuffix='60').join(pd.read_csv('sumario_120.csv').iloc[2:].set_index(['cod_neg']), rsuffix='120').join(pd.read_csv('sumario_365.csv').iloc[2:].set_index(['cod_neg']), rsuffix='365')
stockMovements = stockMovements.reset_index()
stockMovements = stockMovements.fillna(0)
    
stockMovements_values = stockMovements[stockMovements.columns[2:]]
stockMovements_cias = stockMovements[stockMovements.columns[1:2]]

#print(pd.DataFrame(columns=stockMovements_cias.T[stockMovements_cias.T.columns[1:]].values, data=stockMovements_values.values()))

#teste.corr()

#sum_corr = teste.corr().sum().sort_values(ascending=True).index.values
#plt.figure(figsize=(13, 8))
#sns.heatmap(correlation(pos_list), annot=True, cmap=”Greens”);


# Create a TSNE instance: model
model = TSNE(learning_rate=200)

# Apply fit_transform to samples: tsne_features
tsne_features = model.fit_transform(seeds.values)

# Select the 0th feature: xs
xs = tsne_features[:,0]

# Select the 1st feature: ys
ys = tsne_features[:,1]

# Scatter plot, coloring by variety_numbers
plt.scatter(xs, ys, c=varietisSeedsNumbers['varietiesNumbers'].tolist())
plt.show()

#A t-SNE map of the stock market
#t-SNE provides great visualizations when the individual samples can be labeled.
# In this exercise, you'll apply t-SNE to the company stock price data.
# A scatter plot of the resulting t-SNE features, labeled by the company names, gives you a map of the stock market!
# The stock price movements for each company are available as the array normalized_movements (these have already been normalized for you).
# The list companies gives the name of each company. PyPlot (plt) has been imported for you.
# Import TSNE
from sklearn.manifold import TSNE

# Create a TSNE instance: model
model = TSNE(learning_rate=50, random_state=17)

# Apply fit_transform to normalized_movements: tsne_features
#normalized_movements = normalize(stockMovementsT.values)
tsne_features = model.fit_transform(stockMovements_values)

print(stockMovements_values)

# Select the 0th feature: xs
#A t-SNE map of the stock market
xs = tsne_features[:,0]

# Select the 1th feature: ys
ys = tsne_features[:,1]


# Scatter plot
plt.scatter(xs, ys, alpha=0.5, edgecolor='none', s=40, cmap=plt.cm.get_cmap('nipy_spectral', 10))

# Annotate the points
for x, y, company in zip(xs, ys, stockMovements_cias.values):
    plt.annotate(company, (x, y), fontsize=10, alpha=0.75)
plt.show()
