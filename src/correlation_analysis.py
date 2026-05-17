import numpy as np

def compute_correlations(features):

    correlations = []

    for i in range(len(features)-1):

        corr = np.corrcoef(
            features[i],
            features[i+1]
        )[0,1]

        correlations.append(corr)

    return correlations


def detect_forgery(correlations):

    avg_corr = np.mean(correlations)
    print("Average Correlation:", avg_corr)
    if avg_corr < 0.60:
        return "FAKE"

    return "REAL"