from typing import List, Union, Any

class SaliencyList(list):
    """
    A Python list subclass that provides numpy-compatible attributes (.shape, .min(), .max(), __array__)
    while remaining 100% JSON-serializable as a standard list for FastAPI schemas.
    """
    @property
    def shape(self):
        if len(self) == 0:
            return (0,)
        first = self[0]
        if isinstance(first, (list, tuple)):
            return (len(self), len(first))
        return (len(self),)

    def min(self, *args, **kwargs):
        import numpy as np
        return float(np.asarray(list(self)).min(*args, **kwargs))

    def max(self, *args, **kwargs):
        import numpy as np
        return float(np.asarray(list(self)).max(*args, **kwargs))

    def __array__(self, dtype=None):
        import numpy as np
        return np.asarray(list(self), dtype=dtype)

    def __eq__(self, other):
        import numpy as np
        if isinstance(other, (int, float, np.number)):
            return np.asarray(self) == other
        return super().__eq__(other)


def normalize_saliency_matrix(
    matrix: Union[List[List[float]], Any],
    clip_percentile: float = 99.5
) -> SaliencyList:
    """
    Normalizes a 2D saliency matrix to [0, 1] range.
    Supports both numpy ndarray and native Python list-of-lists.
    Returns SaliencyList (list compatible with .shape, .min(), .max()).
    """
    try:
        import numpy as np
        arr = np.array(matrix, dtype=np.float32)
        if arr.size == 0:
            return SaliencyList(matrix)
        upper_val = np.percentile(arr, clip_percentile)
        arr = np.clip(arr, 0.0, upper_val)
        min_val, max_val = float(arr.min()), float(arr.max())
        if max_val - min_val > 1e-8:
            norm_arr = (arr - min_val) / (max_val - min_val)
        else:
            norm_arr = np.zeros_like(arr)
        rows = [SaliencyList([round(float(v), 4) for v in row]) for row in norm_arr]
        return SaliencyList(rows)
    except ImportError:
        # Pure Python fallback
        flat = [val for row in matrix for val in row]
        if not flat:
            return SaliencyList(matrix)
        min_val, max_val = min(flat), max(flat)
        if max_val - min_val > 1e-8:
            rows = [SaliencyList([round(float((v - min_val) / (max_val - min_val)), 4) for v in row]) for row in matrix]
        else:
            rows = [SaliencyList([0.0 for _ in row]) for row in matrix]
        return SaliencyList(rows)
