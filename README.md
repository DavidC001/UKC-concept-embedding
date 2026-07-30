# Semantics over Structure 
Large Language Models (LLMs) acquire world knowledge implicitly through textual patterns, resulting in latent representations that are opaque, difficult to inspect, and challenging to edit. 
This work investigates whether curated Knowledge Graphs (KGs) can serve as a controlled, editable semantic layer to ground these models. Using the Universal Knowledge Core (UKC), we generate Knowledge Graph Embeddings (KGEs) and test their ability to replicate the geometric structures, such as hierarchical orthogonality, found in LLM representations. 
To distinguish between genuine semantic alignment and mere high-dimensional artifacts, we introduce a "scrambled" version of the UKC that preserves graph statistics while destroying semantic relations. Our results demonstrate that while KGEs can successfully recreate the geometric properties of both true and scrambled worlds, the alignment between KGEs and LLM embeddings depends on the semantic integrity of the underlying graph. 
These findings suggest that curated knowledge bases can act as a valid "semantic shadow" of the world, offering a principled path toward more controllable and steerable language models.

## Repository Structure
```
anlp/
├── data_build/        # creating the UKC triplets dataset
├── ukc_embedding/     # training KGE models on the UKC dataset
├── geometry/          # evaluating hierarchical structure in embeddings
├── mapping/           # training mapper between sentence and concept embeddings
```

### How to use

##### 1. Generate UKC triplets dataset

```bash
# normalize and create the UKC dataset
python data_build/normalize_concept_hierarchy.py

# to generate UKC_original dataset
python data_build/triplets_cut_split_original.py 
# or to generate UKC_control dataset
python data_build/triplets_cut_split_control.py 

cd ukc_embedding

# rm data/UKC dir
rm -r data/UKC
# move ../dataset/UKC to data/UKC
mv ../dataset/UKC data/UKC

source set_env.sh
# then run ukc_embedding/datasets/process.py to process it
python datasets/process.py
```

##### 2. Train KGE model on UKC dataset

```bash
cd ukc_embedding
sh set_env.sh
python run.py
```

##### 3. Evaluation of the hierarchy in the concept embeddings

```bash
python geometry/run.py
```

##### 4. Training mapper MLP between gloss embedding and concept embedding

```bash
cd mapping
python main.py
```


### Notice

The datasets used in this project are not uploaded to the repository
