"""Base Knowledge Graph embedding model."""
from abc import ABC, abstractmethod

import torch
from torch import nn


class KGModel(nn.Module, ABC):
    """Base Knowledge Graph Embedding model class.

    Attributes:
        sizes: Tuple[int, int, int] with (n_entities, n_relations, n_entities)
        rank: integer for embedding dimension
        sym: torch.Tensor with indices of symmetric relations (for inverse relation handling)
        dropout: float for dropout rate
        gamma: torch.nn.Parameter for margin in ranking-based loss
        bias: string for whether to learn or fix bias (none for no bias)
        init_size: float for embeddings' initialization scale
        entity: torch.nn.Embedding with entity embeddings
        rel: torch.nn.Embedding with relation embeddings
        bh: torch.nn.Embedding with head entity bias embeddings
        bt: torch.nn.Embedding with tail entity bias embeddings
    """

    def __init__(
            self, 
            sizes, rank, sym,
            dropout, gamma, bias, init_size
            ):
        """Initialize KGModel."""
        super(KGModel, self).__init__()
        
        self.sizes = sizes
        self.rank = rank
        self.dropout = dropout
        self.bias = bias
        self.init_size = init_size
        self.data_type = torch.float  # tables are float32 (subclasses initialise with this dtype)
        self.gamma = nn.Parameter(torch.Tensor([gamma]), requires_grad=False)
        self.entity = nn.Embedding(sizes[0], rank)
        self.rel = nn.Embedding(sizes[1], rank)
        self.bh = nn.Embedding(sizes[0], 1)
        self.bh.weight.data = torch.zeros((sizes[0], 1))
        self.bt = nn.Embedding(sizes[0], 1)
        self.bt.weight.data = torch.zeros((sizes[0], 1))
        # relations that are their own inverse: queried with the same relation in both directions
        self.sym = torch.as_tensor(list(sym) if sym is not None else [], dtype=torch.long)

    @abstractmethod
    def get_queries(self, queries):
        """Compute embedding and biases of queries.

        Args:
            queries: torch.LongTensor with query triples (head, relation, tail)
        Returns:
             lhs_e: torch.Tensor with queries' embeddings (embedding of head entities and relations)
             lhs_biases: torch.Tensor with head entities' biases
        """
        pass

    @abstractmethod
    def get_rhs(self, queries, eval_mode):
        """Get embeddings and biases of target entities.

        Args:
            queries: torch.LongTensor with query triples (head, relation, tail)
            eval_mode: boolean, true for evaluation, false for training
        Returns:
             rhs_e: torch.Tensor with targets' embeddings
                    if eval_mode=False returns embedding of tail entities (n_queries x rank)
                    else returns embedding of all possible entities in the KG dataset (n_entities x rank)
             rhs_biases: torch.Tensor with targets' biases
                         if eval_mode=False returns biases of tail entities (n_queries x 1)
                         else returns biases of all possible entities in the KG dataset (n_entities x 1)
        """
        pass

    @abstractmethod
    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        """Compute similarity scores or queries against targets in embedding space.

        Args:
            lhs_e: torch.Tensor with queries' embeddings
            rhs_e: torch.Tensor with targets' embeddings
            eval_mode: boolean, true for evaluation, false for training
        Returns:
            scores: torch.Tensor with similarity scores of queries against targets
        """
        pass

    def score(self, lhs, rhs, eval_mode):
        """Scores queries against targets

        Args:
            lhs: Tuple[torch.Tensor, torch.Tensor] with queries' embeddings and head biases
                 returned by get_queries(queries)
            rhs: Tuple[torch.Tensor, torch.Tensor] with targets' embeddings and tail biases
                 returned by get_rhs(queries, eval_mode)
            eval_mode: boolean, true for evaluation, false for training
        Returns:
            score: torch.Tensor with scores of queries against targets
                   if eval_mode=True, returns scores against all possible tail entities, shape (n_queries x n_entities)
                   else returns scores for triples in batch (shape n_queries x 1)
        """
        lhs_e, lhs_biases = lhs
        rhs_e, rhs_biases = rhs
        score = self.similarity_score(lhs_e, rhs_e, eval_mode)
        if self.bias == 'constant':
            return self.gamma.item() + score
        elif self.bias == 'learn':
            if eval_mode:
                return lhs_biases + rhs_biases.t() + score
            else:
                return lhs_biases + rhs_biases + score
        else:
            return score

    def get_factors(self, queries):
        """Computes factors for embeddings' regularization.

        Args:
            queries: torch.LongTensor with query triples (head, relation, tail)
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor] with embeddings to regularize
        """
        head_e = self.entity(queries[:, 0])
        rel_e = self.rel(queries[:, 1])
        rhs_e = self.entity(queries[:, 2])
        return head_e, rel_e, rhs_e

    def forward(self, queries, eval_mode=False):
        """KGModel forward pass.

        Args:
            queries: torch.LongTensor with query triples (head, relation, tail)
            eval_mode: boolean, true for evaluation, false for training
        Returns:
            predictions: torch.Tensor with triples' scores
                         shape is (n_queries x 1) if eval_mode is false
                         else (n_queries x n_entities)
            factors: embeddings to regularize
        """
        # get embeddings and similarity scores
        lhs_e, lhs_biases = self.get_queries(queries)
        # queries = F.dropout(queries, self.dropout, training=self.training)
        rhs_e, rhs_biases = self.get_rhs(queries, eval_mode)
        # candidates = F.dropout(candidates, self.dropout, training=self.training)
        predictions = self.score((lhs_e, lhs_biases), (rhs_e, rhs_biases), eval_mode)

        # get factors for regularization
        factors = self.get_factors(queries)
        return predictions, factors

    def get_ranking(self, queries, filters, batch_size=1000):
        """Compute filtered ranking of correct entity for evaluation.

        Args:
            queries: torch.LongTensor with query triples (head, relation, tail)
            filters: filters[(head, relation)] gives entities to ignore (filtered setting)
            batch_size: int for evaluation batch size

        Returns:
            ranks: torch.Tensor with ranks or correct entities
        """
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            b_begin = 0
            candidates = self.get_rhs(queries, eval_mode=True)
            while b_begin < len(queries):
                these_queries = queries[b_begin:b_begin + batch_size].cuda()

                q = self.get_queries(these_queries)
                rhs = self.get_rhs(these_queries, eval_mode=False)

                scores = self.score(q, candidates, eval_mode=True)
                targets = self.score(q, rhs, eval_mode=False)

                # set filtered and true scores to -1e6 to be ignored
                for i, query in enumerate(these_queries):
                    filter_out = filters[(query[0].item(), query[1].item())]
                    filter_out += [queries[b_begin + i, 2].item()]
                    scores[i, torch.LongTensor(filter_out)] = -1e6
                ranks[b_begin:b_begin + batch_size] += torch.sum(
                    (scores >= targets).float(), dim=1
                ).cpu()
                b_begin += batch_size
        return ranks


    def _get_outlier_info(self, examples, all_ranks, rank_threshold, dataset):
        """
        Processes ranks to identify outlier triples and checks their frequency.
        """
        outlier_details = []
        
        # --- 1. Calculate Frequencies from Training Examples (including inverse relations) ---
        train_examples = dataset.get_examples("train")
        
        # Calculate entity frequencies (uses the processed KGE IDs as keys)
        all_entities = torch.cat([train_examples[:, 0], train_examples[:, 2]]).cpu()
        ent_ids, ent_counts = all_entities.unique(return_counts=True)
        ent_freqs = {i.item(): c.item() for i, c in zip(ent_ids, ent_counts)}
        
        # Calculate relation frequencies (uses the processed KGE IDs as keys)
        all_relations = train_examples[:, 1].cpu()
        rel_ids, rel_counts = all_relations.unique(return_counts=True)
        rel_freqs = {i.item(): c.item() for i, c in zip(rel_ids, rel_counts)}

        num_original_relations = dataset.n_predicates // 2
        
        # --- 2. Process RHS Outliers (h, r, ?) ---
        q_rhs = examples.clone() 
        ranks_rhs = all_ranks['rhs']
        rhs_outliers_idx = (ranks_rhs >= rank_threshold).nonzero(as_tuple=True)[0]
        
        for idx in rhs_outliers_idx:
            h_id, r_id, t_id = q_rhs[idx].tolist()
            rank = ranks_rhs[idx].item()
            
            # --- START: ID to NAME Conversion ---
            h_name = dataset.id_to_entity.get(h_id, f"ID:{h_id}")
            t_name = dataset.id_to_entity.get(t_id, f"ID:{t_id}")
            
            # Map inverse relation ID back to original relation ID for lookup
            original_r_id = r_id
            is_inverse = False
            if r_id >= num_original_relations:
                original_r_id = r_id - num_original_relations
                is_inverse = True

            r_name_base = dataset.id_to_relation.get(original_r_id, f"ID:{original_r_id}")
            r_display = f"{r_name_base}" + (" (INV)" if is_inverse else "")
            # --- END: ID to NAME Conversion ---

            # Frequencies are looked up using the KGE IDs (h_id, r_id, t_id)
            h_freq = ent_freqs.get(h_id, 0)
            r_freq = rel_freqs.get(r_id, 0)
            t_freq = ent_freqs.get(t_id, 0)
            
            outlier_details.append(
                f"RHS Outlier (h, r, ?): Rank {int(rank)} for triple ({h_name}, {r_display}, {t_name})"
                f" | Frequencies (in training set): h={h_freq}, r={r_freq}, t={t_freq}"
            )
            
        # --- 3. Process LHS Outliers (?, r, t) ---
        q_lhs = examples.clone() 
        ranks_lhs = all_ranks['lhs']
        lhs_outliers_idx = (ranks_lhs >= rank_threshold).nonzero(as_tuple=True)[0]
        
        for idx in lhs_outliers_idx:
            h_id, r_id, t_id = q_lhs[idx].tolist()
            rank = ranks_lhs[idx].item()
            
            # --- START: ID to NAME Conversion ---
            h_name = dataset.id_to_entity.get(h_id, f"ID:{h_id}")
            t_name = dataset.id_to_entity.get(t_id, f"ID:{t_id}")
            
            original_r_id = r_id
            is_inverse = False
            if r_id >= num_original_relations:
                original_r_id = r_id - num_original_relations
                is_inverse = True

            r_name_base = dataset.id_to_relation.get(original_r_id, f"ID:{original_r_id}")
            r_display = f"{r_name_base}" + (" (INV)" if is_inverse else "")
            # --- END: ID to NAME Conversion ---

            h_freq = ent_freqs.get(h_id, 0)
            r_freq = rel_freqs.get(r_id, 0)
            t_freq = ent_freqs.get(t_id, 0)
            
            outlier_details.append(
                f"LHS Outlier (?, r, t): Rank {int(rank)} for triple ({h_name}, {r_display}, {t_name})"
                f" | Frequencies (in training set): h={h_freq}, r={r_freq}, t={t_freq}"
            )
            
        return outlier_details

    def compute_metrics(self, examples, filters, batch_size=500):
        """Compute ranking-based evaluation metrics.
    
        Args:
            examples: torch.LongTensor of size n_examples x 3 containing triples' indices
            filters: Dict with entities to skip per query for evaluation in the filtered setting
            batch_size: integer for batch size to use to compute scores

        Returns:
            Evaluation metrics (mean rank, mean reciprocical rank and hits)
        """
        mean_rank = {}
        mean_reciprocal_rank = {}
        hits_at = {}
        median = {}
        mean_rank_90 = {}
        all_ranks = {}            # ← ADD THIS

        for m in ["rhs", "lhs"]:
            q = examples.clone()
            if m == "lhs":
                tmp = torch.clone(q[:, 0])
                q[:, 0] = q[:, 2]
                q[:, 2] = tmp
                inverse = ~torch.isin(q[:, 1], self.sym.to(q.device)) if self.sym is not None and len(self.sym) else torch.ones_like(q[:, 1], dtype=torch.bool)
                q[inverse, 1] += self.sizes[1] // 2
            ranks = self.get_ranking(q, filters[m], batch_size=batch_size)

            all_ranks[m] = ranks.cpu()      # ← ADD THIS
            mean_rank[m] = torch.mean(ranks).item()
            median[m] = torch.median(ranks).item()
            
            mean_reciprocal_rank[m] = torch.mean(1. / ranks).item()

            k = int(0.9 * len(ranks))
            sorted_ranks = torch.sort(ranks).values
            mean_rank_90[m] = torch.mean(sorted_ranks[:k]).item()

            hits_at[m] = torch.FloatTensor((list(map(
                lambda x: torch.mean((ranks <= x).float()).item(),
                (1, 3, 10)
            ))))

        return mean_rank, mean_reciprocal_rank, hits_at, median, mean_rank_90, all_ranks
