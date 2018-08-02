from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.feature_extraction import text
stop = text.ENGLISH_STOP_WORDS

articles = pd.read_csv('../../DataSets/articles/bbc-text.csv')
#pat = r'\b(?:{})\b'.format('|'.join(stop))
#articles['text'] = articles['text'].str.replace(pat, '')

# Import NMF
from sklearn.decomposition import NMF

# Create a TfidfVectorizer: tfidf
tfidf = TfidfVectorizer()

#documents = ['cats say meowee ', 'dogs say woof', 'dogs chase cats']
documents = articles['text'].tolist()

category = articles['category'].tolist()

# Apply fit_transform to document: csr_mat
csr_mat = tfidf.fit_transform(documents)

print(csr_mat.shape)

# Print result of toarray() method
#print(csr_mat.toarray())

# Get the words: words
words = tfidf.get_feature_names()

#NMF applied to Wikipedia articles
# In the video, you saw NMF applied to transform a toy word-frequency array.
# Now it's your turn to apply NMF, this time using the tf-idf word-frequency array of Wikipedia articles,
# given as a csr matrix articles. Here, fit the model and transform the articles.
# In the next exercise, you'll explore the result.

# Create an NMF instance: model
model = NMF(n_components=5)

# Fit the model to articles
model.fit(csr_mat)

# Transform the articles: nmf_features
nmf_features = model.transform(csr_mat)

# Print the NMF features
print(nmf_features)

#NMF features of the Wikipedia articles
##Now you will explore the NMF features you created in the previous exercise.
# A solution to the previous exercise has been pre-loaded, so the array nmf_features is available.
# Also available is a list titles giving the title of each Wikipedia article.
# When investigating the features, notice that for both actors, the NMF feature 3
# has by far the highest value. This means that both articles are reconstructed using mainly
# the 3rd NMF component. In the next video, you'll see why: NMF components represent topics (for instance, acting!).

# Create a pandas DataFrame: df
df = pd.DataFrame(nmf_features, index=category)

# Print the row for 'Anne Hathaway'
print(df.loc['entertainment'])

# Print the row for 'Denzel Washington'
print(df.loc['politics'])

#NMF learns topics of documents
# In the video, you learned when NMF is applied to documents, the components correspond to
# topics of documents, and the NMF features reconstruct the documents from the topics. Verify this for yourself for the NMF
# model that you built earlier using the Wikipedia articles. Previously, you saw that the 3rd NMF feature value was
# high for the articles about actors Anne Hathaway and Denzel Washington. In this exercise, identify the topic of the corresponding NMF component.
# The NMF model you built earlier is available as model, while words is a list of the words that label the columns of the word-frequency array.
# After you are done, take a moment to recognise the topic that the articles about Anne Hathaway and Denzel Washington have in common!

# Create a DataFrame: components_df
components_df = pd.DataFrame(model.components_, columns=words)
print(components_df)

# Print the shape of the DataFrame
print(components_df.shape)

# Select row 3: component
component = components_df.iloc[4]

# Print result of nlargest
print(component.nlargest())
