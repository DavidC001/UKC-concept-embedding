"""Training utils."""
import datetime
import os


def get_savedir(model, dataset, rank):
    """Get unique saving directory name."""
    dt = datetime.datetime.now()
    date = dt.strftime("%m_%d")
    save_dir = os.path.join(
        os.environ["LOG_DIR"], date, dataset,
        model + dt.strftime('_%H_%M_%S')
    )
    os.makedirs(save_dir)
    return save_dir


def scalar_metrics(metrics):
    """Metrics dict -> {name: float}, dropping the per-query all_ranks."""
    return {k: float(v) for k, v in metrics.items() if k != "all_ranks"}


def avg_both(mrs, mrrs, hits, medians, mean_rank_90s, all_rankss):
    """Aggregate metrics for missing lhs and rhs.

    Args:
        mrs: Dict[str, float]
        mrrs: Dict[str, float]
        hits: Dict[str, torch.FloatTensor]

    Returns:
        Dict[str, torch.FloatTensor] mapping metric name to averaged score
    """
    mr = (mrs['lhs'] + mrs['rhs']) / 2.
    mrr = (mrrs['lhs'] + mrrs['rhs']) / 2.
    h = (hits['lhs'] + hits['rhs']) / 2.
    h1 = h[0]
    h2 = h[1]
    h3 = h[2]
    median = (medians['lhs'] + medians['rhs']) / 2.
    mean_rank_90 = (mean_rank_90s['lhs'] + mean_rank_90s['rhs']) / 2.

    # all_ranks is NOT averaged — we return both lists
    # ranks_lhs = all_ranks["lhs"].tolist()
    # ranks_rhs = all_ranks["rhs"].tolist()
    # return {'mean_rank': mr, 'mrr': mrr, 'hits@[1,3,10]': h, 'hits@1': h1, 'hits@3': h2, 'hits@10': h3}
    return {'mean_rank': mr, 'mrr': mrr, 'hits@1': h1, 'hits@3': h2, 'hits@10': h3, 'median': median, 'mean_rank_90': mean_rank_90, 'all_ranks': all_rankss}


def format_metrics(metrics, split):
    """Format metrics for logging."""
    result = "\t {} mean_rank: {:.2f} | ".format(split, metrics['mean_rank'])
    result += "mrr: {:.3f} | ".format(metrics['mrr'])
    # result += "hits@1: {:.3f} | ".format(metrics['hits@[1,3,10]'][0])
    # result += "hits@3: {:.3f} | ".format(metrics['hits@[1,3,10]'][1])
    # result += "hits@10: {:.3f}".format(metrics['hits@[1,3,10]'][2])
    result += "hits@1: {:.3f} | ".format(metrics['hits@1'])
    result += "hits@3: {:.3f} | ".format(metrics['hits@3'])
    # result += "hits@10: {:.3f}".format(metrics['hits@10'])
    result += "hits@10: {:.3f} | ".format(metrics['hits@10'])
    result += "mean_rank_90: {:.3f} | ".format(metrics['mean_rank_90'])
    # median += "median: {:.3f}".format(metrics['median'])
    result += "median: {:.3f}".format(metrics['median'])
    return result


def count_params(model):
    """Count total number of trainable parameters in model"""
    total = 0
    for x in model.parameters():
        if x.requires_grad:
            res = 1
            for y in x.shape:
                res *= y
            total += res
    return total
