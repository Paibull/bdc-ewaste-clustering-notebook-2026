import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from .config import ACTIVE_MODEL, FUSION_DIM, K, MODEL_PRESETS, SEED, VIEW_DIM


def l2(values):
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


class FusedKMeans(ClusterMixin, BaseEstimator):
    def __init__(
        self,
        k=K,
        seed=SEED,
        preset=ACTIVE_MODEL,
        normalize_views=False,
        view_dim=VIEW_DIM,
        fusion_dim=FUSION_DIM,
    ):
        self.k = k
        self.seed = seed
        self.preset = preset
        self.normalize_views = normalize_views
        self.view_dim = view_dim
        self.fusion_dim = fusion_dim
        self.view_pca = {}
        self.fusion_pca = None
        self.model = None

    def _join(self, views, fit=False):
        blocks = []
        for name in MODEL_PRESETS[self.preset]:
            values = l2(views[name])
            if fit:
                components = min(self.view_dim, min(values.shape) - 1)
                self.view_pca[name] = PCA(
                    n_components=components,
                    svd_solver="randomized",
                    whiten=False,
                    random_state=0,
                )
                block = self.view_pca[name].fit_transform(values)
            else:
                block = self.view_pca[name].transform(values)
            if self.normalize_views:
                block = l2(block)
            blocks.append(block.astype(np.float32, copy=False))
        return l2(np.concatenate(blocks, axis=1))

    def fit(self, views, y=None):
        joined = self._join(views, fit=True)
        components = min(self.fusion_dim, min(joined.shape) - 1)
        self.fusion_pca = PCA(
            n_components=components,
            svd_solver="randomized",
            whiten=False,
            random_state=0,
        )
        space = l2(self.fusion_pca.fit_transform(joined))
        self.model = KMeans(
            n_clusters=self.k, n_init=20, random_state=self.seed
        ).fit(space)
        self.labels_ = self.model.labels_
        return self

    def fit_predict(self, views, y=None):
        return self.fit(views, y).labels_

    def transform(self, views):
        joined = self._join(views)
        return l2(self.fusion_pca.transform(joined))

    def predict(self, views):
        return self.model.predict(self.transform(views))

    def nearest_centroid(self, views):
        space = self.transform(views)
        centers = l2(self.model.cluster_centers_)
        return np.argmax(space @ centers.T, axis=1)
