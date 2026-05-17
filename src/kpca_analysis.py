from sklearn.decomposition import KernelPCA
import numpy as np

def apply_kpca(features):

    kpca = KernelPCA(
        n_components=25,
        kernel='rbf'
    )

    reduced_features = kpca.fit_transform(features)

    return reduced_features