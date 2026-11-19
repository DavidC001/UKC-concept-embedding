import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch import nn


class KGOptimizer:
    """Knowledge Graph embedding model optimizer.

    KGOptimizers performs loss computations with negative sampling and gradient descent steps.

    Attributes:
        model: models.base.KGModel
        regularizers: {name: regularizers.Regularizer}, all added to the loss (see build_regularizers)
        optimizer: torch.optim.Optimizer
        batch_size: An integer for the training batch size
        neg_sample_size: An integer for the number of negative samples
        double_neg: A boolean (True to sample both head and tail entities)
    """

    def __init__(
            self, 
            model, regularizers, optimizer,
            batch_size, neg_sample_size, double_neg, verbose=True
            ):
        """Inits KGOptimizer."""
        self.model = model
        self.regularizers = dict(regularizers)
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.verbose = verbose
        self.double_neg = double_neg
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
        self.neg_sample_size = neg_sample_size
        self.n_entities = model.sizes[0]

    def reduce_lr(self, factor=0.8):
        """Reduce learning rate.

        Args:
            factor: float for the learning rate decay
        """
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= factor

    def get_neg_samples(self, input_batch):
        """Sample negative examples.

        Args:
            input_batch: torch.LongTensor of shape (batch_size x 3) with ground truth training triples

        Returns:
            negative_batch: torch.Tensor of shape (neg_sample_size x 3) with negative examples
        """
        negative_batch = input_batch.repeat(self.neg_sample_size, 1)
        batch_size = input_batch.shape[0]
        negsamples = torch.Tensor(np.random.randint(
            self.n_entities,
            size=batch_size * self.neg_sample_size)
        ).to(input_batch.dtype)
        negative_batch[:, 2] = negsamples
        if self.double_neg:
            negsamples = torch.Tensor(np.random.randint(
                self.n_entities,
                size=batch_size * self.neg_sample_size)
            ).to(input_batch.dtype)
            negative_batch[:, 0] = negsamples
        return negative_batch

    def neg_sampling_loss(self, input_batch):
        """Compute KG embedding loss with negative sampling.

        Args:
            input_batch: torch.LongTensor of shape (batch_size x 3) with ground truth training triples.

        Returns:
            loss: torch.Tensor with negative sampling embedding loss
            factors: torch.Tensor with embeddings weights to regularize
        """
        # positive samples
        positive_score, factors = self.model(input_batch)
        positive_score = F.logsigmoid(positive_score)

        # negative samples 
        neg_samples = self.get_neg_samples(input_batch)
        negative_score, _ = self.model(neg_samples)
        negative_score = F.logsigmoid(-negative_score)
        loss = - torch.cat([positive_score, negative_score], dim=0).mean()
        return loss, factors

    def no_neg_sampling_loss(self, input_batch):
        """Compute KG embedding loss without negative sampling.

        Args:
            input_batch: torch.LongTensor of shape (batch_size x 3) with ground truth training triples

        Returns:
            loss: torch.Tensor with embedding loss
            factors: torch.Tensor with embeddings weights to regularize
        """
        predictions, factors = self.model(input_batch, eval_mode=True)
        truth = input_batch[:, 2]
        log_prob = F.logsigmoid(-predictions)
        idx = torch.arange(0, truth.shape[0], dtype=truth.dtype)
        pos_scores = F.logsigmoid(predictions[idx, truth]) - F.logsigmoid(-predictions[idx, truth])
        log_prob[idx, truth] += pos_scores
        loss = - log_prob.mean()
        return loss, factors

    def calculate_loss(self, input_batch):
        """Compute KG embedding loss and regularization loss.

        Args:
            input_batch: torch.LongTensor of shape (batch_size x 3) with ground truth training triples

        Returns:
            loss: torch.Tensor with embedding loss plus every regularizer
            reg_terms: {name: torch.Tensor} with each regularizer's (weighted) term, detached
        """
        if self.neg_sample_size > 0:
            loss, factors = self.neg_sampling_loss(input_batch)
        else:
            predictions, factors = self.model(input_batch, eval_mode=True)
            truth = input_batch[:, 2]
            loss = self.loss_fn(predictions, truth)

        reg_terms = {}
        for name, reg in self.regularizers.items():
            term = reg(factors, self.model)
            loss = loss + term
            reg_terms[name] = term.detach()
        return loss, reg_terms

    def _accumulate(self, totals, loss, reg_terms):
        totals["loss"] = totals.get("loss", 0.0) + float(loss)
        for name, term in reg_terms.items():
            totals[name] = totals.get(name, 0.0) + float(term)

    def calculate_valid_loss(self, examples):
        """Compute KG embedding loss over validation examples.

        Args:
            examples: torch.LongTensor of shape (N_valid x 3) with validation triples

        Returns:
            loss: float, total loss averaged over the validation batches
            reg_terms: {name: float} with each regularizer's term averaged over the batches
        """
        b_begin = 0
        totals, counter = {}, 0
        with torch.no_grad():
            while b_begin < examples.shape[0]:
                input_batch = examples[
                              b_begin:b_begin + self.batch_size
                              ].cuda()
                b_begin += self.batch_size
                self._accumulate(totals, *self.calculate_loss(input_batch))
                counter += 1
        means = {k: v / counter for k, v in totals.items()}
        return means.pop("loss"), means

    def epoch(self, examples):
        """Runs one epoch of training KG embedding model.

        Args:
            examples: torch.LongTensor of shape (N_train x 3) with training triples

        Returns:
            loss: float, total loss averaged over the training batches
            reg_terms: {name: float} with each regularizer's term averaged over the batches
        """
        actual_examples = examples[torch.randperm(examples.shape[0]), :]
        totals, counter = {}, 0
        with tqdm.tqdm(total=examples.shape[0], unit='ex', disable=not self.verbose) as bar:
            bar.set_description(f'train loss')
            b_begin = 0
            while b_begin < examples.shape[0]:
                input_batch = actual_examples[
                              b_begin:b_begin + self.batch_size
                              ].cuda()

                # gradient step
                l, reg_terms = self.calculate_loss(input_batch)
                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()

                b_begin += self.batch_size
                self._accumulate(totals, l.detach(), reg_terms)
                counter += 1
                bar.update(input_batch.shape[0])
                bar.set_postfix(loss=f'{l.item():.4f}')

        means = {k: v / counter for k, v in totals.items()}
        return means.pop("loss"), means
