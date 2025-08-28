import pathlib, joblib, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

class Vectorizer:
    def __init__(self, models_dir: str, n_components: int = 256):
        self.models_dir = pathlib.Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.vec_path = self.models_dir / "tfidf.joblib"
        self.svd_path = self.models_dir / "svd.joblib"
        self.vectorizer = None
        self.svd = None
        self.n_components = n_components

    def fit(self, texts: list[str]):
        self.vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=60000)
        X = self.vectorizer.fit_transform(texts)
        k = min(self.n_components, min(X.shape)-1)
        self.svd = TruncatedSVD(n_components=k)
        Z = self.svd.fit_transform(X)
        joblib.dump(self.vectorizer, self.vec_path)
        joblib.dump(self.svd, self.svd_path)
        return self._l2(Z)

    def load(self):
        self.vectorizer = joblib.load(self.vec_path)
        self.svd = joblib.load(self.svd_path)

    def transform(self, texts: list[str]) -> np.ndarray:
        if self.vectorizer is None or self.svd is None:
            self.load()
        X = self.vectorizer.transform(texts)
        Z = self.svd.transform(X)
        return self._l2(Z)

    @staticmethod
    def _l2(Z):
        Z = np.asarray(Z)
        n = np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9
        return Z / n
