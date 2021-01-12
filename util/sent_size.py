import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize

df = pd.read_csv('../datasets/articles-validation/validation-sentences.csv', dtype=np.dtype('str'))
print(df.describe())

df['value_tokens'] = df['value'].apply(word_tokenize).apply(lambda x: len(x))
df['value_char'] = df['value'].str.len()
df['sentence_tokens'] = df['sentence'].apply(word_tokenize).apply(lambda x: len(x))

print(df.groupby(['property']).mean())
