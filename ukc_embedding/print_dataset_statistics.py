# open file in /data/UKC/train
# this file contains a list of triplets in the format: concept1, relation, concept2
# we want to count the number of unique concepts and relations in this file
# and for each relation, count the number of triplets that contain that relation

from collections import defaultdict
from pathlib import Path

def print_dataset_statistics(file_path: Path) -> None:
    unique_concepts = set()
    relation_counts = defaultdict(int)

    with file_path.open() as f:
        for line in f:
            concept1, relation, concept2 = line.strip().split('\t')
            unique_concepts.add(concept1)
            unique_concepts.add(concept2)
            relation_counts[relation] += 1

    print(f"Number of unique concepts: {len(unique_concepts)}")
    print("Relation counts:")
    for relation, count in relation_counts.items():
        print(f"{relation}: {count}")
        
if __name__ == "__main__":
    dataset_path = Path("data/UKC/train")
    print_dataset_statistics(dataset_path)