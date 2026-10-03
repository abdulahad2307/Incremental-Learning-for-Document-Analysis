import matplotlib.pyplot as plt

def plot_openset_histograms(y_true, y_pred, scores, savepath=None, known_mask=None):
    """
    Plots histogram of EVM scores for known and unknown true classes.
    known_mask: bool per sample (True = class known to the EVM), e.g. evm_openset_metrics(...)["is_known"].
    """
    if known_mask is None:
        known_mask = [yt != "unknown" for yt in y_true]
    known = []
    unknown = []
    for is_known, s in zip(known_mask, scores):
        if not is_known:
            unknown.append(s)
        else:
            known.append(s)
    plt.hist(known, bins=30, alpha=0.6, label="Known true")
    plt.hist(unknown, bins=30, alpha=0.6, label="Unknown true")
    plt.xlabel("EVM max probability / confidence")
    plt.ylabel("Frequency")
    plt.title("EVM Prediction Scores: Known vs Unknown Samples")
    plt.legend()
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=120)
        print(f"Saved open set histogram to {savepath}")
    else:
        plt.show()
    plt.close()
