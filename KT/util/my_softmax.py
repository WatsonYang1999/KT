import numpy as np


def softmax(logits):
    """
    Compute the softmax activation for a set of logits.

    Arguments:
    logits : array-like
        The input logits. It can be a 1D array (vector) or a 2D array (matrix).

    Returns:
    probs : array-like
        The output probabilities after applying the softmax activation.
    """
    # Check if the input is a vector or a matrix
    if len(logits.shape) > 1:
        # Matrix case: subtract the maximum value from each row for numerical stability
        max_logits = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    else:
        # Vector case: subtract the maximum value for numerical stability
        max_logit = np.max(logits)
        exp_logits = np.exp(logits - max_logit)
        probs = exp_logits / np.sum(exp_logits)

    return probs.astype(np.float64).tolist()

if __name__ == '__main__':
    print(softmax(np.array([5,4,3,2])))
    print(softmax(np.array([4,5,2,1])))
