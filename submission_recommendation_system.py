# -*- coding: utf-8 -*-
"""Submission: Recommendation System.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Mr_kTNRCzoqmCOpixp2xlitGdSvBU7z-

Load Dataset
"""

!unzip /content/shared_articles.csv.zip

!unzip /content/users_interactions.csv.zip

import pandas as pd

shared_articles = pd.read_csv("/content/shared_articles.csv")
users_interactions = pd.read_csv("/content/users_interactions.csv")

print("Total shared article", len(shared_articles.contentId))
print("Total user interaction with articles", len(users_interactions.personId))

"""Exploratory Data for Shared Article"""

shared_articles.head(3)

shared_articles.info()

import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(shared_articles['eventType'])

"""Because of 'content remove' is mean that the article content is not available, then 'content remove' will be dropped from the dataset"""

shared_article = shared_articles[shared_articles['eventType'] == 'CONTENT SHARED']
shared_article.head(3)

sns.countplot(shared_articles['lang'])

shared_articles.lang.value_counts()

shared_articles[shared_articles.lang.isin(['la', 'es', 'ja'])].text.tolist()

"""Because of la, es, and ja language is rare overall of the all language, then it will be dropped"""

shared_article = shared_article[shared_articles.lang.isin(['en', 'pt'])]
shared_article.head(3)

"""Exploratory Data for Users Interaction"""

users_interactions.head(5)

users_interactions.info()

sns.countplot(users_interactions['eventType'])

"""Because of the differences interaction types of users, then we will ordering it according to the most high interaction type with the most high value, in this case 'comment created' is the most high interaction"""

event_typeStrength = {'VIEW':1.0,
                  'LIKE':2.0,
                  'BOOKMARK':3.0,
                  'FOLLOW':4.0,
                  'COMMENT CREATED':5.0}
users_interactions['eventStrength'] = users_interactions['eventType'].apply(lambda x: event_typeStrength[x])
users_interactions.head(3)

users_interactions.userRegion.value_counts()

users_interactions.userCountry.value_counts()

"""Data Preprocessing

Because of the system will be hard to make recommendation (from the user preference) for users that only have a little of interaction with the article, then we will limit it with only users with > 3 will be proceed

users_interactionCount to count the length of the row in the data according to each personId with its contentId (count how much content that interact with user). 

users_interactionLimit to count user_interactionCount with only > 3 interaction
"""

users_interactionsCount = users_interactions.groupby(['personId', 'contentId']).size().groupby('personId').size()
print('all users: ', len(users_interactionsCount))
users_interactionsLimit = users_interactionsCount[users_interactionsCount > 3].reset_index()[['personId']]
print('users with at least 4 interactions: ', len(users_interactionsLimit))

print('all interactions: ', len(users_interactions))
interactions_usersSelected = users_interactions.merge(users_interactionsLimit,
                                                      how = 'right',
                                                      left_on = 'personId',
                                                      right_on = 'personId')
print('interactions with at least 4 interactions: ', len(interactions_usersSelected))

users_interactions.head()

"""import nltk
nltk.download('stopwords')"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Merge the two datasets
merged_data = pd.merge(users_interactions, shared_article, on='contentId')

# Create the user-item matrix
user_item_matrix = merged_data.pivot_table(index='personId', columns='contentId', values='eventStrength', aggfunc='mean').fillna(0)

print(merged_data)

"""Model development: Content Based Filtering"""

# Compute the cosine similarity between the item-item matrix and the user-item matrix
item_item_similarity = cosine_similarity(user_item_matrix.T)
user_item_similarity = cosine_similarity(user_item_matrix)

# Make recommendations
def recommend_articles(user_id, item_item_similarity, user_item_matrix):
    item_index = user_item_matrix.columns
    user_vector = user_item_matrix.loc[user_id].values.reshape(-1, 1)
    scores = item_item_similarity.dot(user_vector)
    scores = pd.DataFrame(scores, index=item_index, columns=['score'])
    scores = scores.sort_values('score', ascending=False)
    return scores.index[:10].tolist()

def get_article_info(article_id, article_data):
    """Retrieve the title and text of an article given its ID."""
    title = article_data.loc[article_id]['title']
    text = article_data.loc[article_id]['text']
    return title, text

article_data = shared_article

# Set the index of the article data DataFrame to the article IDs
article_data.set_index('contentId', inplace=True)

# Recommend articles for a specific user
user_id = 344280948527967603
recommended_articles = recommend_articles(user_id, item_item_similarity, user_item_matrix)

# Print the title and text of each recommended article
print(f"{recommended_articles}")
for article_id in recommended_articles:
    title, text = get_article_info(article_id, article_data)
    print(f"Title: {title}")

"""Evaluating model with precision and recall"""

# Use user_id same as the previous cell
user_id = 344280948527967603
recommended_articles = recommend_articles(user_id, item_item_similarity, user_item_matrix)

# Get the ground truth articles for the user
truth_articles = users_interactions[users_interactions.personId == user_id]['contentId'].tolist()
truth_articles

# Compute precision and recall
true_positives = len(set(recommended_articles).intersection(truth_articles))
precision = true_positives / len(recommended_articles)
recall = true_positives / len(truth_articles)
print(f"Precision score: {precision} \n Recall score: {recall}")