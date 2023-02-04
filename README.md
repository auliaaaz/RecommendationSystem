# Report

## Project Overview

DeskDrop is a campaign launched by Google to communicate with its internal employees. It provides a feature for sharing relevant articles with employees. The dataset used in this project is a 12-month log from DeskDrop, which is a real dataset. The purpose of this project is to create a recommendation system. Before proceeding, it's important to understand that a recommendation system is a set of software tools and techniques that provide suggestions for items that may be useful to a user (the definition is based on [Research Paper OnRcommendation System](https://www.globalscientificjournal.com/researchpaper/Research_Paper_On_Recommendation_System.pdf)).

The purpose of this project is to recommend articles to the user based on their interactions with the articles. The solution approach used in this project is a content-based filtering model. Content-based recommenders provide recommendations by comparing the representation of the content describing an item or a product to the representation of the content describing the user's interests (the user's profile of interest). According to the research paper  [Application of Content-Based Approach in Research Paper Recommendation System for a Digital Library](https://www.researchgate.net/publication/287427252_Application_of_Content-Based_Approach_in_Research_Paper_Recommendation_System_for_a_Digital_Library) content-based recommendations learn a profile of the new user's interests based on the features present in the objects that the user has rated (in this case, based on user interactions). The method used in this project is the cosine similarity method. The importance of this project is to help users filter and predict only those articles that are most likely to be watched by the user. A content-based recommendation will help these users make more accurate article recommendations.

## Business Understanding

### Problem Statements

The problem is : How does the content-based model attempt to filter and predict the most relevant articles based on the articles that users interact with using the cosine similarity approach?

### Goals

Provide articles recommendation which most relevan to users profile with filtering and predicting those articles that users most interact with using top-n recommendation to users.

### Solution statements
- Working with Exploratory Data Analysis (EDA) for every dataset, because we're using two dataset and preprocessing the data.
- Using content-based recommendation method to recommended articles to user, in this case using cosine similarity method.
- Evaluating the model that using cosine similarity method with precision and recall.

## Data Understanding

The dataset used in this project is from the DeskDrop platform developed by CI&T. The dataset provided is based on real data collected over a 12-month period from March 2016 to February 2017 and can be accessed on [Kaggle](https://www.kaggle.com/datasets/gspmoreira/articles-sharing-reading-from-cit-deskdrop). It contains information about 73,000 logged users interactions on over 3,000 public articles shared on the platform.

This dataset features some distinctive characteristics:
- timestamp : is the time when event has occured.
- eventType (shared articles data) : is the conditions of the article either it removed (Content Removed) nor available (Content Shared).
- contentId (shared articles and users interactions data): is Id (unique number) of articles.
- authorPersonId : is ID (unique number) of article's author.
- authorSessionId : is session ID of the author, different session indicate different session of author created the articles.
- authorUserAgent : is type of browser the author used.
- authorRegion : is region of author.
- authorCountry : is country of author.
- contentType : is format articles.
- url : is URL of the articles.
- title : is title of the articles.
- text : is content of articles.
- lang : is language that articles used.
- eventType (users interactions data): is type of interactions that user make with articles.
- personId: is ID of users.
- sessionId: is session ID of users.
- userAgent: is type of browser that users used.
- userRegion: is region of users.
- userCountry: is country of users.

### Exploratory Data Analysis (EDA)
The first dataset that will be analyze is articles shared data. Because of there are some of data type in that data is categorical data, then some of that will be shown as visualization below.

Figure 1
![Screenshot 2023-02-01 144941](https://user-images.githubusercontent.com/91725987/216051321-637ea64d-ddc1-4825-b16b-c2c5d5274a8d.jpg)

As depicted in Figure 1, the value of 'content shared' is higher than that of 'content removed.' This is because 'content removed' refers to articles that are no longer available and thus have been dropped from the data, leaving only the available articles.

Figure 2
![Screenshot 2023-02-01 145020](https://user-images.githubusercontent.com/91725987/216051375-1fdccb9a-aad9-460c-9ee6-a5ccec09e193.jpg)

As depicted in Figure 2, the majority of articles are written in the English (en) and Portuguese (pt) languages, while the usage of other languages is rare. As a result, we will only consider English and Portuguese languages in the data.

Next, we move on to the second dataset, which is the user interaction data. Like the first dataset, some of the categorical data will be visualized.

Figure 3
![Screenshot 2023-02-01 145222](https://user-images.githubusercontent.com/91725987/216051407-b5f9d58e-d1d7-4ac2-8c51-585307966e26.jpg)

As depicted in Figure 3, the most common type of user interaction is 'View', which means that users interact by viewing articles. Since there are 5 types of interactions, they will be scored based on the user's level of interaction with the data and assigned to a new column (eventStrength). The score represents the strength of the interaction, with 1 representing the weakest interaction (View) and 4 representing the strongest interaction (1: View, 2: Like, 3: Bookmark, 4: Comment Created).

## Data Preparation

- The first step is to filter the user interactions. In this step, only users with 4 or more interactions will be processed, as it will be difficult for the system to make relevant recommendations if the user does not have enough information (interactions with articles). The process involves counting how many contents interact with users.
- Next, the two datasets, the user interactions and shared interaction data, will be merged according to 'contentId' to connect the type of user interactions and the content of the articles.

## Modeling
The project utilizes a content-based filtering model for making article recommendations to the user based on their interactions with the articles. This approach compares the representation of content that describes an item to the representation of the content that describes the user's profile of interest. As per the research paper, "Application of Content-Based Approach in Research Paper Recommendation System for a Digital Library" content-based recommendations learn a profile of the user's interests based on the features present in the objects the user has rated.

In this project, the cosine similarity method is used for the content-based filtering. This method measures the similarity between two sequences, in this case the 'user-item matrix' and the 'item-item matrix' (which is the transpose of the user-item matrix), using the 'eventStrength' to represent the interactions between a user and an article. The 'eventStrength' represents the mean value of the interactions between a person and the content they interacted with. The formula for the cosine similarity is shown below.

$$ CosineSimilarity(A, B) =  ((A * B) \over (||A|| * ||B||)

Where:

- A and B are the vectors being compared
- (A * B) is the dot product of the two vectors, which measures the similarity between two vectors based on the angle between them
- ||A|| and ||B|| are the magnitudes (or lengths) of the two vectors

The next step is to make recommendations for a specific user. This is done by providing the user ID and the item-item similarity matrix and user-item matrix to the model. The function then calculates a score for each article by multiplying the user vector with the item-item similarity matrix and dividing it by the sum of the absolute values of each row in the item-item similarity matrix. The scores are then sorted in descending order, and the top 10 articles with the highest scores are returned as the recommended articles for the user.

```
# Make recommendations
def recommend_articles(user_id, item_item_similarity, user_item_matrix):
    item_index = user_item_matrix.columns
    user_vector = user_item_matrix.loc[user_id].values.reshape(-1, 1)
    scores = item_item_similarity.dot(user_vector)
    scores = pd.DataFrame(scores, index=item_index, columns=['score'])
    scores = scores.sort_values('score', ascending=False)
    return scores.index[:10].tolist()
```

The code4 below is show the top-10 articles recommendation to a user with userID: 344280948527967603

```
[4876769046116846438, 310515487419366995, -908052164352446106, 3569727790804487273, 8254285966695461849, -5784991738549272379, -1068603220639552685, 521382866757596943, -3723217532224917485, -4974757204495953627]
Title: Shopping em BH terá fazenda urbana de 2.700 m²
Title: 71 erros de português que precisam sumir dos seus e-mails
Title: IoT a favor do relacionamento médico-paciente
Title: DualShock 4 Repair
Title: Dr. consulta: uma revolução no setor da saúde
Title: NodeMCU (ESP8266) o módulo que desbanca o Arduino e facilitará a Internet das Coisas...
Title: Onde a tecnologia e o cuidado se encontram? - Saúde Business
Title: Optimize Arduino Memory Usage
Title: 11 Internet of Things (IoT) Protocols You Need to Know About
Title: Alecrim/AlecrimCoreData
```
The values in array is represent the contentId of each articles.

## Evaluation

The evalution metrics used in this project are precision and recall metrics. Precision is represent how many of the articles recommended to the user are actually relevant to the user. Recall is represent the proportion of relevant articles that the recommendation system is able to find for the user. The formulas of precision and recall used in this recommendation system are:

$$ Precision = {NumberOfRelevantItemsRetrieved \over NumberOfRetrievedItems} $$

$$ Recall = {NumberOfRelevantItemsRetrieved \over NumberOfRelevanItemsInDataset} $$

Precision and recall score after testing the model with recommend articles to a user are shown below. 

![Screenshot 2023-02-01 150810](https://user-images.githubusercontent.com/91725987/216051429-2f4e363b-1f43-47fd-8fcf-f671a46b196e.jpg)

A precision score of 0.8 means that out of all the items that the recommendation model has recommended, 80% of them were relevant to the user, while 20% were not. On the other hand, a recall score of 0.32 means that out of all the relevant items that exist for a user, the recommendation model was able to recommend only 32% of them, while 68% of them were not recommended.

## Conclusions

According to the evaluation scores, the content-based filtering model with the cosine similarity method can be used to make article recommendations to users. But, especially to the recall score, 0.32 indicates that the model is not very good at capturing all the relevant items that exist for the user. So with that, the model need to improving for example with another modelling method.

---This is the end of the report---


