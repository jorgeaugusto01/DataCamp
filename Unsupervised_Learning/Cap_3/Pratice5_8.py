# A tf-idf word-frequency array
# In this exercise, you'll create a tf-idf word frequency array for a toy collection of documents.
# For this, use the TfidfVectorizer from sklearn. It transforms a list of documents into a word frequency array,
# which it outputs as a csr_matrix. It has fit() and transform() methods like other sklearn objects.
# You are given a list documents of toy documents about pets. Its contents have been printed in the IPython Shell.

# Import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
articles = pd.read_csv('../../DataSets/articles/bbc-text.csv')

# Create a TfidfVectorizer: tfidf
tfidf = TfidfVectorizer()

#documents = ['cats say meowee ', 'dogs say woof', 'dogs chase cats']
documents = articles['text'].tolist()
# Apply fit_transform to document: csr_mat
csr_mat = tfidf.fit_transform(documents)

print(csr_mat.shape)

# Print result of toarray() method
#print(csr_mat.toarray())

# Get the words: words
words = tfidf.get_feature_names()
#print(len(words))

# Print words
#print(words)

# Clustering Wikipedia part I
# You saw in the video that TruncatedSVD is able to perform PCA on sparse arrays in csr_matrix format,
# such as word-frequency arrays. Combine your knowledge of TruncatedSVD and k-means to cluster some popular pages from Wikipedia.
# In this exercise, build the pipeline. In the next exercise, you'll apply it to the word-frequency array of some Wikipedia articles.
# Create a Pipeline object consisting of a TruncatedSVD followed by KMeans.
# (This time, we've precomputed the word-frequency matrix for you, so there's no need for a TfidfVectorizer).
# The Wikipedia dataset you will be working with was obtained from here.

# Perform the necessary imports
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline

with pd.option_context('display.max_rows', 3000, 'display.max_columns', 3000):
    print(articles.loc[443])

# Create a TruncatedSVD instance: svd
svd = TruncatedSVD(n_components=50)

# Create a KMeans instance: kmeans
kmeans =  KMeans(n_clusters=5)

# Create a pipeline: pipeline
pipeline =  make_pipeline(svd, kmeans)

# Fit the pipeline to articles
pipeline.fit(csr_mat)

# Calculate the cluster labels: labels
labels = pipeline.predict(csr_mat)

print(labels)
# Create a DataFrame aligning labels and titles: df
df = pd.DataFrame({'label': labels, 'article': articles['category']})

with pd.option_context('display.max_rows', 3000, 'display.max_columns', 100):
    # Display df sorted by cluster label
    print(df.sort_values('article'))

