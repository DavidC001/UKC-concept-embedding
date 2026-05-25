# Knowledge-Based Concept Embeddings
In this paper, we explored the development of knowledge-based concept token embeddings to bridge the gap between structured knowledge and LLM representations. We then evaluated the embeddings to verify their ability to reconstruct lexical relations and to assess the feasibility of integrating them with LLMs. The results are promising, ensuring the potential for expanding this methodology.

###### This repository consist of 5 sub projects

```
anlp/
├── data_build/        # creating the UKC triplets dataset
├── ukc_embedding/     # training KGE models on the UKC dataset
├── geometry/          # evaluating hierarchical structure in embeddings
├── mapping/           # training an MLP mapper between sentence and concept embeddings
├── WSD/               # fine-tuning a model for the WSD task
└── assets/            # visualization assets
```

### How to use

##### 1. Generate UKC triplets dataset
HERE WE CAN SAY THAT THE UKC DATA WE NEED TO GIVE

```bash
# First run data_build/triplets_cut_split.py to create the UKC dataset
python data_build/triplets_cut_split.py

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

##### 5. Training and evaluating a WSD

```bash
python -m WSD.main --mode "experiments"
```

### Notice

The datasets used in this project is not uploaded to the repository