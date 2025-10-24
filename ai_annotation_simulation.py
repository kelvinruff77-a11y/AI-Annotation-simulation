
# AI Annotation Simulation Project

import pandas as pd
from collections import Counter

# Load dataset
df = pd.read_csv('ai_annotation_dataset.csv')

# Inspect data
print("Sample data:")
print(df.head())

# Calculate annotation agreement (simple majority vote)
def majority_vote(row):
    votes = [row['Annotator_1'], row['Annotator_2'], row['Annotator_3']]
    return Counter(votes).most_common(1)[0][0]

df['Final_Label'] = df.apply(majority_vote, axis=1)

# Save results
df.to_csv('ai_annotation_results.csv', index=False)

print("\nAnnotation simulation complete!")
print(df[['Text', 'Final_Label']])
