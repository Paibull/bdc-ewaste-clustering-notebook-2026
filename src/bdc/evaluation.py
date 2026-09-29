import numpy as np
from sklearn.metrics import adjusted_rand_score


class ResearchMetrics:
    def seed_stability(self, runs, threshold=0.95):
        count = len(runs)
        scores = np.eye(count, dtype=np.float32)
        for left in range(count):
            for right in range(left + 1, count):
                scores[left, right] = scores[right, left] = adjusted_rand_score(
                    runs[left], runs[right]
                )
        means = (scores.sum(axis=1) - 1) / max(count - 1, 1)
        medoid = int(np.argmax(means))
        values = scores[:, medoid]
        rng = np.random.default_rng(20260928)
        null = np.asarray([
            adjusted_rand_score(runs[medoid], rng.permutation(runs[medoid]))
            for _ in range(100)
        ])
        observed = float(np.median(values))
        return {
            "medoid": medoid,
            "mean_ari": float(values.mean()),
            "median_ari": float(np.median(values)),
            "runs_above_threshold": int(np.count_nonzero(values >= threshold)),
            "null_95": float(np.quantile(null, 0.95)),
            "null_p": float((1 + np.count_nonzero(null >= observed)) / 101),
        }

    def strict_transfer(self, truth, prediction):
        truth = np.asarray(truth, dtype=str)
        prediction = np.asarray(prediction, dtype=object)
        mapped = np.asarray([
            value is not None and str(value).strip() not in ("", "nan", "-1", "unmapped")
            for value in prediction
        ])
        recall = {
            family: float(np.mean(prediction[truth == family] == family))
            for family in np.unique(truth)
        }
        return {
            "micro_top1": float(np.mean(prediction == truth)),
            "macro_top1": float(np.mean(list(recall.values()))),
            "mapped_coverage": float(mapped.mean()),
            "recall": recall,
        }

    def incremental_r2(self, target, baseline, full):
        target = np.asarray(target, dtype=np.float64)
        base_error = np.square(target - np.asarray(baseline)).sum()
        full_error = np.square(target - np.asarray(full)).sum()
        return float((base_error - full_error) / base_error)

    def material_detection(self, truth, scores, names):
        from sklearn.metrics import average_precision_score, roc_auc_score

        truth = np.asarray(truth)
        scores = np.asarray(scores)
        return {
            name: {
                "auroc": float(roc_auc_score(truth[:, index], scores[:, index])),
                "average_precision": float(
                    average_precision_score(truth[:, index], scores[:, index])
                ),
            }
            for index, name in enumerate(names)
        }
