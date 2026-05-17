import numpy as np

def compute_correlations(features):

    correlations = []

    for i in range(len(features)-1):

        corr = np.corrcoef(
            features[i],
            features[i+1]
        )[0,1]

        if not np.isnan(corr):
            correlations.append(corr)

    return correlations


def detect_forgery(correlations):

    correlations = np.array(correlations)

    correlations = correlations[
        ~np.isnan(correlations)
    ]

    median_corr = np.median(correlations)

    print("Median Correlation:", median_corr)

    if median_corr < 0.59:
        return "FAKE"

    return "REAL"