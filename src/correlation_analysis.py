def detect_forgery(correlations):

    avg_corr = np.mean(correlations)

    if avg_corr < 0.75:
        return "FAKE"

    return "REAL"