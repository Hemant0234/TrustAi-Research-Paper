import math
from typing import Dict, Tuple, List, Union

def compute_pairwise_spatial_correlation(
    map_a: Union[List[List[float]], any],
    map_b: Union[List[List[float]], any]
) -> float:
    """
    Computes spatial Pearson correlation coefficient between two 2D saliency maps.
    Returns value normalized into [0, 1] range (1 = identical spatial attribution).
    """
    flat_a = [float(v) for row in map_a for v in row]
    flat_b = [float(v) for row in map_b for v in row]

    n = len(flat_a)
    if n == 0 or len(flat_b) != n:
        return 0.5

    mean_a = sum(flat_a) / n
    mean_b = sum(flat_b) / n

    diff_a = [x - mean_a for x in flat_a]
    diff_b = [y - mean_b for y in flat_b]

    var_a = sum(x * x for x in diff_a)
    var_b = sum(y * y for y in diff_b)

    if var_a < 1e-8 or var_b < 1e-8:
        return 0.5

    cov = sum(x * y for x, y in zip(diff_a, diff_b))
    corr = cov / math.sqrt(var_a * var_b)
    
    # Scale from [-1, 1] to [0, 1]
    scaled_corr = (corr + 1.0) / 2.0
    return max(0.0, min(1.0, scaled_corr))

def compute_agreement_and_disagreement_maps(
    maps: List[List[List[float]]],
    weights: List[float]
) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Computes spatial Agreement Map and Disagreement Map across N explainers.
    """
    if not maps:
        return [], []
        
    num_maps = len(maps)
    h = len(maps[0])
    w = len(maps[0][0]) if h > 0 else 0

    total_w = sum(weights)
    norm_weights = [wt / total_w for wt in weights] if total_w > 1e-8 else [1.0 / num_maps] * num_maps

    agreement_map = [[0.0 for _ in range(w)] for _ in range(h)]
    disagreement_map = [[0.0 for _ in range(w)] for _ in range(h)]

    for r in range(h):
        for c in range(w):
            vals = [maps[i][r][c] for i in range(num_maps)]
            weighted_mean = sum(v * wt for v, wt in zip(vals, norm_weights))
            weighted_var = sum(wt * ((v - weighted_mean) ** 2) for v, wt in zip(vals, norm_weights))
            
            disagreement_map[r][c] = weighted_var
            agreement_map[r][c] = weighted_mean

    # Normalize disagreement map to [0, 1]
    flat_dis = [v for row in disagreement_map for v in row]
    max_dis = max(flat_dis) if flat_dis else 0.0
    if max_dis > 1e-8:
        disagreement_map = [[round(v / max_dis, 4) for v in row] for row in disagreement_map]
    else:
        disagreement_map = [[0.0 for _ in row] for row in disagreement_map]

    # Agreement map is weighted_mean * (1 - disagreement)
    for r in range(h):
        for c in range(w):
            agreement_map[r][c] = agreement_map[r][c] * (1.0 - disagreement_map[r][c])

    flat_agr = [v for row in agreement_map for v in row]
    max_agr = max(flat_agr) if flat_agr else 0.0
    if max_agr > 1e-8:
        agreement_map = [[round(v / max_agr, 4) for v in row] for row in agreement_map]
    else:
        agreement_map = [[0.0 for _ in row] for row in agreement_map]

    from app.fusion.normalization import SaliencyList
    agr_sl = SaliencyList([SaliencyList(row) for row in agreement_map])
    dis_sl = SaliencyList([SaliencyList(row) for row in disagreement_map])
    return agr_sl, dis_sl
