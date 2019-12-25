#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


des = pd.read_csv('destinations.csv')


# ## From the previous work we noticed that all popular machine learning models and neural network gave a very low accuracy. We will try different method to improve the accuracy.

# In[7]:


dest_small[:4]


# ### The destination.csv file consists of all the scrh_destination_id and latent description of destination. We will try to extract the important features using principal component analysis
# ### The destination_csv has 150 columns and we will extract only 3 important features.

# In[5]:


from sklearn.decomposition import PCA  

pca = PCA(n_components=3)  
dest_small = pca.fit_transform(des[["d{0}".format(i+1)                                    for i in range(149)]])
dest_small = pd.DataFrame(dest_small)  
dest_small["srch_destination_id"]= des["srch_destination_id"]


# ### The work is done using the entire training and test set. Training set is around 4gb with 37670293 rows and the test set is around 1 gb with 2 million rows.

# In[8]:


import pandas  as pd
  
train =pd.read_csv("train.csv")


# In[10]:


train["date_time"] = pd.to_datetime(train["date_time"]) 
train["year"] = train["date_time"].dt.year  
train["month"] = train["date_time"].dt.month


# In[11]:


train.corr()["hotel_cluster"]


# ### Aggregating on srch_destination_id will find the most popular hotel clusters for each destination.We'll be then able to predict that a user who searches for a destination is going to one of the most popular hotel clusters for that destination.We can first generate scores for each hotel_cluster in each srch_destinaation_id .We'll weight bookings higher than clicks.This is because the test data is all booking data , and this is what we want to predict. We want to include click information,but downweight it to reflect this. Step by step,we'll:
# 1)Group train by srch_destination_id and hotel_cluster\
# 2)Iterate through each group\
# 3)Assign 1 point to each hotel cluster where is_booking is true.\
# 4)Assign .15 points to each hotel cluster where is_booking is false.\
# 5)Assign the score to the srch_destination_id/ hotel_cluster combination in a dictionary
# 
# 

# In[13]:


def make_key(items):  
    return "-".join([str(i) for i in items])


# In[14]:


match_cols = ["srch_destination_id"]  
clusters_cols = match_cols + ['hotel_cluster']


# In[15]:


groups = train.groupby(clusters_cols)


# ### We'll next want to transform this dictionary to find the top 5 hotel clusters for each srch_destination_id.In order to this,we'll: Loop through each key in top_clusters Find the top 5 clusters for that key. Assign the top 5 clusters to a new dictionary ,cluster_dict.

# In[ ]:


top_clusters = {}  
for name,group in groups:  
    clicks = len(group.is_booking[group.is_booking == False])
    bookings = len(group.is_booking[group.is_booking == True]) 
    score = bookings + .15*clicks  
    clus_name = make_key(name[:len(match_cols)])
    print(clus_name)
    if clus_name not in top_clusters:
        top_clusters[clus_name] = {} 
    top_clusters[clus_name][name[-1]] = score


# In[22]:


test = pd.read_csv('test.csv')
test.head()


# In[17]:


import operator  
cluster_dict = {}  

for n in top_clusters:
    tc = top_clusters[n] 
    top = [l[0] for l in sorted(tc.items(),
                                key=operator.itemgetter(1),
                                reverse=True)[:5]]
    cluster_dict[n]=top


# # Predictions
# ### Once we know the top clusters for each srch_destination_id ,we can quickly make predictions .To make predictions,all we have to do is: Iterate through each row in test. Extract the srch_destination_id for the row. Find the top clusters for that destination id. Append the top clusters to preds

# In[23]:


preds = [] 
for index,row in test.iterrows():
    key= make_key(row[m] for m in match_cols)  
    if key in cluster_dict:
        preds.append(cluster_dict[key]) 
    else:
        preds.append([])


# ### There was a post that details a data leak which allows you to match users in the training set from the testing set using a set of columns including user_location_country ,and user_location_region. The post link is at the end of this notebook. We'll use the information from the post to match users from the testing set back to the training set, which will boost our score.
# ### The first step is to find users in the training set that match users in the testing set. In order to do this,we need to: split the training data into groups based on the match columns. Loop through the testing data. Create an index based on the match columns. Get any matches betweeb the testing data and the training data using the groups.

# In[25]:


match_cols = ['user_location_country' , 'user_location_region', 
              'user_location_city','hotel_market',
              'orig_destination_distance']

#groups =t1.groupby(match_cols)  
groups =train.groupby(match_cols) 

def generate_exact_matches(row,match_cols):
    index = tuple([row[t] for t in match_cols])
    try:
        group = groups.get_group(index)
    except Exception:
        return []
    clus = list(set(group.hotel_cluster))
    return clus

exact_matches = []
for i in range(test.shape[0]):
    exact_matches.append(generate_exact_matches(test.iloc[i],
                                                match_cols))


# In[27]:


most_common_clusters=list(train.hotel_cluster.value_counts().
                          head().index)


# In[37]:


predictions =[most_common_clusters 
              for i in range(test.shape[0])]
predictions


# All predictions are same hence did not take the full print out

# ## Final Predictions
# 
# ### Combine exact_matches,preds, and most_common_clusters. Only take the unique predictions,in sequential order,using the f5 functionEnsure we have a maximum of 5 predictions for each row in the testing set.

# In[32]:


def f5(seq, idfun=None): 
    if idfun is None:
        def idfun(x): return x
    seen = {}
    result = []
    for item in seq:
        marker = idfun(item)
        if marker in seen: continue
        seen[marker] = 1
        result.append(item)
    return result
    
full_preds = [f5(exact_matches[p] + preds[p] +
                 most_common_clusters)[:5]
              for p in range(len(preds))]
#metrics.mapk([[l] for l in test["hotel_cluster"]], full_preds, k=5)


# In[33]:


write_p = [" ".join([str(l) for l in p]) for p in full_preds]
write_frame = ["{0},{1}".format(test["id"][i], write_p[i])
               for i in range(len(full_preds))]
write_frame = ["id,hotel_cluster"] + write_frame
with open("predictions.csv", "w+") as f:
    f.write("\n".join(write_frame))


# In[34]:


data2 = pd.read_csv('predictions.csv')


# In[35]:


data2.head()


# In[36]:


full_preds


# In[ ]:




