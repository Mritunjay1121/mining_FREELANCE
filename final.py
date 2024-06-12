import pandas as pd
import numpy as np
import pyfpgrowth
import sys


# def mine_frequent_patterns(input_path, min_support=0.2, min_confidence=0.6):
#     # Load dataset
#     df = pd.read_csv(input_path)
#     l={}

#     items=list(df['itemDescription'].value_counts().index)
#     for i in range(len(items)):
#         l[items[i]]=f"T{i}"
#     user_id=[]
#     for val in df['itemDescription']:
#         user_id.append(l[val])
#     df['userId']=user_id
    
#     transactions = df.groupby('userId')['itemDescription'].apply(list).tolist()
#     patterns = pyfpgrowth.find_frequent_patterns(transactions, 2)  # Adjust support threshold as needed
#     return patterns

# Example usage



# def mine_frequent_patterns(input_path, min_support=2):
#     # Load dataset
#     df = pd.read_csv(input_path)
    
#     # Create a dictionary to map each item to a unique transaction ID
#     item_to_tid = {item: f"T{i}" for i, item in enumerate(df['itemDescription'].unique())}
    
#     # Map each item to its corresponding transaction ID
#     df['userId'] = df['itemDescription'].map(item_to_tid)
    
#     # Group items by userId and create transactions
#     transactions = df.groupby('userId')['itemDescription'].apply(list).tolist()
    
#     # Find frequent patterns using FP-Growth
#     patterns = pyfpgrowth.find_frequent_patterns(transactions, min_support)  # Adjust support threshold as needed
#     return patterns



sys.setrecursionlimit(10000)

def mine_frequent_patterns(input_path, min_support=2):
    # Load dataset
    df = pd.read_csv(input_path)

    # Create a dictionary to map each item to a unique transaction ID
    item_to_tid = {item: f"T{i}" for i, item in enumerate(df['itemDescription'].unique())}

    # Map each item to its corresponding transaction ID
    df['userId'] = df['itemDescription'].map(item_to_tid)

    # Group items by userId and create transactions
    transactions = df.groupby('userId')['itemDescription'].apply(list).tolist()

    # Find frequent patterns using FP-Growth
    patterns = pyfpgrowth.find_frequent_patterns(transactions, min_support)
    
    return patterns


def collaborative_filtering(path, patterns=None):
    
    
    dataset = pd.read_csv(path)
    l={}

    items=list(dataset['itemDescription'].value_counts().index)
    for i in range(len(items)):
        l[items[i]]=f"T{i}"
    user_id=[]
    for val in dataset['itemDescription']:
        user_id.append(l[val])
    dataset['userId']=user_id

    if patterns:
        # Use patterns from Part 1
        frequent_items = list(set(item for pattern in patterns.keys() for item in pattern))
    else:
        # Use unique items from the dataset
        frequent_items = dataset['itemDescription'].unique().tolist()

    # Create a user-item matrix for collaborative filtering
    user_item_matrix = pd.crosstab(dataset['userId'], dataset['itemDescription'])
    
    # Score recommendations based on common items with the user
    recommendations = {}
    for user in user_item_matrix.index:
        user_items = user_item_matrix.loc[user]
        user_recommendations = []
        for item in frequent_items:
            if item not in user_items or user_items[item] == 0:
                user_recommendations.append((item, 1))  # Adding a dummy score of 1 for now
        recommendations[user] = user_recommendations

    # Example metric: Count of recommendations
    performance_metric = {user: len(recommendations[user]) for user in recommendations}
    
    return recommendations, performance_metric


path=input("Enter the path of dataset to mine frequent patterns : ")

use_patterns = input("Do you want to use patterns from Part 1 for recommendations? (yes/no): ").lower()
if use_patterns == 'yes':
    patterns = mine_frequent_patterns(input_path=path)
else:
    patterns = None

recommendations, performance_metric = collaborative_filtering(path=path, patterns=patterns)

# Example output
print("\nRecommendations:")
for user, recs in recommendations.items():
    print(f"User {user}: {recs}")

print("\nPerformance Metric (Count of Recommendations per User):")
print(performance_metric)

# Test dataset for evaluation
test_dataset_path = input("Enter the path to your test dataset for evaluation: ")

_, test_performance_metric = collaborative_filtering(test_dataset_path, patterns)
print("\nPerformance Metric on Test Dataset:")
print(test_performance_metric)

