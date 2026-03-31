'''
This script processes the concept relations from the UKC dataset, filters them based on entity frequency, and splits them into training, validation, and test sets. 
The resulting triplets are saved in a specified directory for use in knowledge graph embedding tasks or other downstream applications.
'''

import pandas as pd
import numpy as np
import os

# Constants
MIN_FREQ = 1
BOOST_FREQ = 1
VALID_RATIO = 0.0
TEST_RATIO = 0.0
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

IS_UKC_ID = 1

RELATION_MAP = {
    20: 'has_hyponym',
    22: 'meronym_has_part',
    34: 'attribute_has_attribute',
    36: 'meronym_has_substance',
    37: 'meronym_has_member',
    43: 'has_aspect'
}

DATA_DIR = "dataset/UKC_CUT_1"

# Load concepts and relations from CSV files
concepts_df = pd.read_csv('dataset/concepts.csv', usecols=['id', 'label'])
relations_df = pd.read_csv('dataset/concept_relations.csv', usecols=['relation_type', 'src_con_id', 'trg_con_id'])


# drop all concepts we don't have a label for in concepts.csv
head_concepts = concepts_df.rename(columns={'id': 'src_con_id', 'label': 'head_label'})
tail_concepts = concepts_df.rename(columns={'id': 'trg_con_id', 'label': 'tail_label'})

triples_df = relations_df.merge(head_concepts, on='src_con_id', how='left')
triples_df = triples_df.merge(tail_concepts, on='trg_con_id', how='left')

triples_df.dropna(subset=['head_label', 'tail_label'], inplace=True)


# rename and order columns to head, relation, tail
final_triples = triples_df[['src_con_id', 'relation_type', 'trg_con_id']]
final_triples.columns = ['head', 'relation', 'tail']


# filter only the triples with more than MIN_FREQ occurrences of head or tail in the dataset
all_entities = pd.concat([final_triples["head"], final_triples["tail"]])
entity_freq = all_entities.value_counts()
entities_to_filter = set(entity_freq[entity_freq < MIN_FREQ].index)

filtered_df = final_triples[~final_triples["head"].isin(entities_to_filter) & ~final_triples["tail"].isin(entities_to_filter)]
n_remove = len(final_triples) - len(filtered_df)

print(f"Removed {n_remove} triples. Remaining triples: {len(filtered_df)}")

# compute number of unique entities in the filtered dataset
unique_entities = set(filtered_df["head"]).union(set(filtered_df["tail"]))
print(f"Number of unique entities in the filtered dataset: {len(unique_entities)}")

# shuffle the triples
filtered_df = filtered_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)


# split into train, valid, test with the given ratios, ensuring that all entities in valid and test are seen in train
print("filtered_df len:", len(filtered_df))
n = len(filtered_df)
n_valid = int(n * VALID_RATIO)
n_test = int(n * TEST_RATIO)

valid_df = filtered_df.iloc[:n_valid]
test_df = filtered_df.iloc[n_valid:n_valid + n_test]
train_df = filtered_df.iloc[n_valid + n_test:]

def move_unseen_to_train(train_df, valid_df, test_df):
    seen = set(train_df["head"]).union(set(train_df["tail"]))

    def process_eval_split(eval_df):
        keep_rows = []
        move_rows = []
        for _, row in eval_df.iterrows():
            h, t = row["head"], row["tail"]
            if h in seen and t in seen:
                keep_rows.append(row)
            else:
                move_rows.append(row)
        return pd.DataFrame(keep_rows), pd.DataFrame(move_rows)

    valid_keep, valid_move = process_eval_split(valid_df)
    test_keep, test_move = process_eval_split(test_df)
    new_train = pd.concat([train_df, valid_move, test_move], ignore_index=True)
    return new_train, valid_keep, test_keep, len(valid_move) + len(test_move)

while True:
    train_df, valid_df, test_df, moved = move_unseen_to_train(train_df, valid_df, test_df)
    if moved == 0:
        break

# boost in training set the relations with concepts that are present less than 5 times in the training set, to ensure they are well represented during training

train_entity_freq = pd.concat([train_df["head"], train_df["tail"]]).value_counts()
rare_entities = set(train_entity_freq[train_entity_freq < BOOST_FREQ].index)

while rare_entities:
    rare_triples = filtered_df[filtered_df["head"].isin(rare_entities) | filtered_df["tail"].isin(rare_entities)]
    train_df = pd.concat([train_df, rare_triples], ignore_index=True)
    
    print(f"Boosted training set with rare triples: {len(rare_triples)}")
    
    train_entity_freq = pd.concat([train_df["head"], train_df["tail"]]).value_counts()
    rare_entities = set(train_entity_freq[train_entity_freq < BOOST_FREQ].index)

os.makedirs(DATA_DIR, exist_ok=True)
train_df.to_csv(f"{DATA_DIR}/train", sep="\t", index=False, header=False)
valid_df.to_csv(f"{DATA_DIR}/valid", sep="\t", index=False, header=False)
test_df.to_csv(f"{DATA_DIR}/test", sep="\t", index=False, header=False)

print("Train:", len(train_df))
print("Valid:", len(valid_df))
print("Test:", len(test_df))
