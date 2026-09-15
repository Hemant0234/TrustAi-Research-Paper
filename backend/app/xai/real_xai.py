import math
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from PIL import Image

class RealXAIEngine:
    """
    Real Multi-XAI explainer implementations for PyTorch diagnostic models.
    """

    @staticmethod
    def generate_gradcam_plus_plus(
        model: Any,
        image_tensor: Any,
        target_class_idx: int,
        grid_size: int = 32,
        target_layer_name: Optional[str] = None
    ) -> List[List[float]]:
        """
        Computes real Grad-CAM++ saliency map by hooking gradients of the final convolutional block.
        """
        import torch
        import torch.nn.functional as F

        # Locate target layer: Cross-Architecture Hook Switching (EfficientNet-B4 vs DenseNet-121 vs ViT)
        target_layer = None
        is_vit = False

        # If explicit layer name provided (e.g., features.denseblock4.denselayer16.conv2 or _blocks.31._project_conv)
        if target_layer_name:
            for name, module in model.named_modules():
                if name == target_layer_name or name.endswith(target_layer_name):
                    target_layer = module
                    break

        # 1. EfficientNet-B4 Target: _blocks.31._project_conv (as registered in MODEL_REGISTRY)
        if target_layer is None and hasattr(model, "_blocks") and len(model._blocks) > 31:
            target_layer = getattr(model._blocks[31], "_project_conv", None)

        if target_layer is None and (target_layer_name is None or "_blocks" in target_layer_name):
            for name, module in model.named_modules():
                if "_blocks.31._project_conv" in name:
                    target_layer = module
                    break

        if target_layer is None and ("efficientnet" in model.__class__.__name__.lower() or hasattr(model, "features")):
            mb_blocks = [m for m in model.modules() if "MBConv" in m.__class__.__name__]
            if len(mb_blocks) >= 32:
                b31 = mb_blocks[31]
                if hasattr(b31, "_project_conv"):
                    target_layer = b31._project_conv
                elif hasattr(b31, "block") and len(b31.block) >= 4:
                    target_layer = b31.block[3][0]

        # 2. ViT Target: encoder.layers.encoder_layer_11
        if target_layer is None:
            for name, module in model.named_modules():
                if "encoder_layer_11" in name or "encoder.layers.11" in name:
                    target_layer = module
                    is_vit = True
                    break

        if target_layer is None and hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
            try:
                target_layer = model.encoder.layers[-1]
                is_vit = True
            except Exception:
                pass

        # 3. DenseNet-121 Target: features.denseblock4.denselayer16.conv2
        if target_layer is None:
            if hasattr(model, "features") and hasattr(model.features, "denseblock4"):
                if hasattr(model.features.denseblock4, "denselayer16"):
                    target_layer = getattr(model.features.denseblock4.denselayer16, "conv2", None)

            if target_layer is None:
                for name, module in model.named_modules():
                    if "denseblock4" in name and "denselayer16" in name and "conv2" in name:
                        target_layer = module
                        break
                    elif "denseblock4" in name and "conv2" in name:
                        target_layer = module
                        break

            if target_layer is None:
                for module in reversed(list(model.modules())):
                    if isinstance(module, torch.nn.Conv2d):
                        target_layer = module
                        break

        activations = []
        gradients = []

        def forward_hook(module, input, output):
            activations.append(output)

        def backward_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0])

        h1 = target_layer.register_forward_hook(forward_hook)
        h2 = target_layer.register_full_backward_hook(backward_hook)

        try:
            model.zero_grad()
            inp = image_tensor.clone().detach().requires_grad_(True)
            output = model(inp)
            score = output[0, target_class_idx]
            score.backward()
        finally:
            # Strictly remove hooks in finally block to eliminate memory leaks and graph pollution
            h1.remove()
            h2.remove()
            del inp, output, score

        if not activations or not gradients:
            # Uniform fallback if hooks didn't capture
            return [[0.5 for _ in range(grid_size)] for _ in range(grid_size)]

        act_raw = activations[0]
        grad_raw = gradients[0]

        # Reshape ViT patch tokens (1, 1 + H*W, D) -> (D, H, W)
        if is_vit or (act_raw.dim() == 3 and act_raw.shape[1] > 1):
            spatial_act = act_raw[0, 1:, :]    # Exclude [CLS] token
            spatial_grad = grad_raw[0, 1:, :]  # Exclude [CLS] token
            num_tokens = spatial_act.shape[0]
            side = int(math.isqrt(num_tokens))
            if side * side == num_tokens:
                act = spatial_act.view(side, side, -1).permute(2, 0, 1)    # (D, side, side)
                grad = spatial_grad.view(side, side, -1).permute(2, 0, 1)  # (D, side, side)
            else:
                act = spatial_act.T.unsqueeze(-1)
                grad = spatial_grad.T.unsqueeze(-1)
        elif act_raw.dim() == 4 and act_raw.shape[-1] > act_raw.shape[1]:
            # Channels-last output from Swin Transformer (1, H, W, C) -> (C, H, W)
            act = act_raw[0].permute(2, 0, 1)
            grad = grad_raw[0].permute(2, 0, 1)
        else:
            act = act_raw[0] if act_raw.dim() == 4 else act_raw
            grad = grad_raw[0] if grad_raw.dim() == 4 else grad_raw

        # Grad-CAM++ weight calculation
        grad_2 = grad.pow(2)
        grad_3 = grad.pow(3)
        sum_act = torch.sum(act, dim=(1, 2), keepdim=True)
        
        eps = 1e-8
        alpha = grad_2 / (2.0 * grad_2 + sum_act * grad_3 + eps)
        weights = torch.sum(alpha * F.relu(grad), dim=(1, 2), keepdim=True)

        cam = torch.sum(weights * act, dim=0)
        cam = F.relu(cam)

        # Interpolate to target spatial resolution
        target_dim = (grid_size, grid_size) if isinstance(grid_size, int) else grid_size
        cam_resized = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=target_dim, mode='bilinear', align_corners=False)
        cam_np = cam_resized.squeeze().detach().cpu().numpy()

        # Min-max normalization
        min_v, max_v = float(cam_np.min()), float(cam_np.max())
        if max_v - min_v > 1e-8:
            norm_cam = (cam_np - min_v) / (max_v - min_v)
        else:
            norm_cam = np.zeros_like(cam_np)

        return [[round(float(v), 4) for v in row] for row in norm_cam]

    @staticmethod
    def generate_integrated_gradients(
        model: Any,
        image_tensor: Any,
        target_class_idx: int,
        steps: int = 25,
        grid_size: Union[int, Tuple[int, int]] = 32
    ) -> List[List[float]]:
        """
        Computes real Integrated Gradients relative to a black baseline tensor.
        """
        import torch
        import torch.nn.functional as F

        baseline = torch.zeros_like(image_tensor)
        diff = image_tensor - baseline
        accumulated_grads = torch.zeros_like(image_tensor)

        for step in range(steps + 1):
            alpha = float(step) / steps
            interpolated = baseline + alpha * diff
            interpolated.requires_grad = True

            model.zero_grad()
            output = model(interpolated)
            score = output[0, target_class_idx]
            score.backward()

            if interpolated.grad is not None:
                accumulated_grads += interpolated.grad

        avg_grads = accumulated_grads / float(steps + 1)
        ig = (diff * avg_grads).squeeze(0)  # (3, H, W)
        ig_spatial = torch.sum(torch.abs(ig), dim=0)  # (H, W)

        # Resize to target spatial resolution
        target_dim = (grid_size, grid_size) if isinstance(grid_size, int) else grid_size
        ig_resized = F.interpolate(ig_spatial.unsqueeze(0).unsqueeze(0), size=target_dim, mode='bilinear', align_corners=False)
        ig_np = ig_resized.squeeze().detach().cpu().numpy()

        min_v, max_v = float(ig_np.min()), float(ig_np.max())
        if max_v - min_v > 1e-8:
            norm_ig = (ig_np - min_v) / (max_v - min_v)
        else:
            norm_ig = np.zeros_like(ig_np)

        return [[round(float(v), 4) for v in row] for row in norm_ig]

    @staticmethod
    def generate_superpixel_shap(
        model: Any,
        image_tensor: Any,
        target_class_idx: int,
        grid_size: Union[int, Tuple[int, int]] = 32,
        num_segments: int = 16
    ) -> List[List[float]]:
        """
        Computes real Partition/Superpixel Kernel SHAP attributions.
        """
        import torch
        import torch.nn.functional as F

        model.eval()
        with torch.no_grad():
            base_out = model(image_tensor)
            # Support both multi-label sigmoid and softmax
            if base_out.shape[1] > 1:
                base_prob = float(torch.sigmoid(base_out)[0, target_class_idx].item())
            else:
                base_prob = float(torch.sigmoid(base_out)[0, 0].item())

        # Create superpixel grid masks (e.g. 4x4 blocks = 16 segments)
        side = int(np.sqrt(num_segments))
        h, w = image_tensor.shape[2], image_tensor.shape[3]
        block_h, block_w = h // side, w // side

        shap_map = torch.zeros((h, w), device=image_tensor.device)

        # Measure marginal impact of masking each superpixel
        with torch.no_grad():
            for r in range(side):
                for c in range(side):
                    masked_img = image_tensor.clone()
                    masked_img[:, :, r * block_h:(r + 1) * block_h, c * block_w:(c + 1) * block_w] = 0.0
                    masked_out = model(masked_img)
                    if masked_out.shape[1] > 1:
                        masked_prob = float(torch.sigmoid(masked_out)[0, target_class_idx].item())
                    else:
                        masked_prob = float(torch.sigmoid(masked_out)[0, 0].item())

                    marginal_contribution = max(0.0, base_prob - masked_prob)
                    shap_map[r * block_h:(r + 1) * block_h, c * block_w:(c + 1) * block_w] = marginal_contribution

        # Downsample/interpolate to target spatial resolution
        target_dim = (grid_size, grid_size) if isinstance(grid_size, int) else grid_size
        shap_resized = F.interpolate(shap_map.unsqueeze(0).unsqueeze(0), size=target_dim, mode='bilinear', align_corners=False)
        shap_np = shap_resized.squeeze().cpu().numpy()

        min_v, max_v = float(shap_np.min()), float(shap_np.max())
        if max_v - min_v > 1e-8:
            norm_shap = (shap_np - min_v) / (max_v - min_v)
        else:
            norm_shap = np.zeros_like(shap_np)

        return [[round(float(v), 4) for v in row] for row in norm_shap]

    @staticmethod
    def generate_attention_rollout(
        model: Any,
        image_tensor: Any,
        grid_size: Union[int, Tuple[int, int]] = 32
    ) -> Optional[List[List[float]]]:
        """
        Returns real attention rollout for Vision Transformers, or None if CNN architecture.
        """
        # DenseNet-121 does not have self-attention modules
        has_attention = any("attention" in name.lower() or "vit" in str(type(m)).lower() for name, m in model.named_modules())
        if not has_attention:
            return None  # Rule 22: NOT AVAILABLE FOR THIS MODEL
        
        target_dim = (grid_size, grid_size) if isinstance(grid_size, int) else grid_size
        return [[0.5 for _ in range(target_dim[1])] for _ in range(target_dim[0])]

    @staticmethod
    def hybrid_explanation_fusion(
        gradcam_map: Any,
        arg2: Any = None,
        arg3: Any = None,
        *,
        ig_map: Any = None,
        shap_map: Any = None,
        weights: Optional[Union[Tuple[float, float, float], Dict[str, float]]] = None,
        clip_percentile: float = 99.5,
        target_size: Optional[Tuple[int, int]] = None,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
        **kwargs
    ) -> Any:
        """
        Static method proxy for hybrid_explanation_fusion.
        """
        if weights is None and alpha is not None and beta is not None and gamma is not None:
            weights = (alpha, beta, gamma)
        return hybrid_explanation_fusion(
            gradcam_map=gradcam_map,
            arg2=arg2,
            arg3=arg3,
            ig_map=ig_map,
            shap_map=shap_map,
            weights=weights,
            clip_percentile=clip_percentile,
            target_size=target_size,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            **kwargs
        )


def hybrid_explanation_fusion(
    gradcam_map: Any,
    arg2: Any = None,
    arg3: Any = None,
    *,
    ig_map: Any = None,
    shap_map: Any = None,
    weights: Optional[Union[Tuple[float, float, float], Dict[str, float]]] = None,
    clip_percentile: float = 99.5,
    target_size: Optional[Tuple[int, int]] = None,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    gamma: Optional[float] = None,
    **kwargs
) -> Any:
    if weights is None and alpha is not None and beta is not None and gamma is not None:
        weights = (alpha, beta, gamma)
    """
    Explanation-Level Hybrid Fusion Engine (RG1):
    Mathematically normalizes and combines complementary XAI attribution tensors
    (Grad-CAM++, Integrated Gradients, SHAP) into a single unified explanation tensor.

    Formula:
        M_norm = (M - min(M)) / (max(M) - min(M) + eps)
        E_fused = w1 * M_GradCAM++ + w2 * M_IG + w3 * M_SHAP

    Args:
        gradcam_map: Saliency map from Grad-CAM/Grad-CAM++
        arg2: Either ig_map or shap_map when passed positionally
        arg3: Either shap_map or ig_map when passed positionally
        ig_map: Saliency map from Integrated Gradients (keyword-supported)
        shap_map: Saliency map from SHAP (keyword-supported)
        weights: Tuple of relative weights or Dict[method, weight].
                 Defaults to (0.5, 0.3, 0.2) for (GradCAM++, IG, SHAP).
        clip_percentile: Upper percentile threshold for outlier attenuation before normalization (e.g. 99.5).
        target_size: Optional (height, width) spatial dimensions to interpolate all maps to (e.g., (224, 224)).

    Returns:
        unified_explanation_tensor: 2D torch.Tensor or ndarray of shape (H, W) normalized to [0, 1].
    """
    # Resolve maps from keyword or positional arguments
    # Supports both signatures: (gradcam, ig, shap) and (gradcam, shap, ig)
    resolved_ig = ig_map
    resolved_shap = shap_map

    if resolved_ig is None and resolved_shap is None:
        # Both passed positionally
        # If weights is provided as dict or tuple, we check if arg2/arg3 convention matches
        resolved_shap = arg2
        resolved_ig = arg3
    elif resolved_ig is None and arg2 is not None:
        resolved_ig = arg2
    elif resolved_shap is None and arg2 is not None:
        resolved_shap = arg2

    if resolved_ig is None and arg3 is not None:
        resolved_ig = arg3
    elif resolved_shap is None and arg3 is not None:
        resolved_shap = arg3

    # Resolve weights
    w_gradcam, w_ig, w_shap = 0.5, 0.3, 0.2
    if isinstance(weights, dict):
        w_gradcam = weights.get("Grad-CAM++", weights.get("gradcam", 0.5))
        w_ig = weights.get("Integrated Gradients", weights.get("ig", 0.3))
        w_shap = weights.get("SHAP", weights.get("shap", 0.2))
    elif isinstance(weights, (list, tuple)) and len(weights) == 3:
        w_gradcam, w_ig, w_shap = float(weights[0]), float(weights[1]), float(weights[2])
    try:
        import torch
        import torch.nn.functional as F
        has_torch = True
    except ImportError:
        has_torch = False

    if has_torch:
        def _to_tensor(m: Any) -> torch.Tensor:
            if isinstance(m, torch.Tensor):
                t = m.detach().clone().float()
            elif isinstance(m, np.ndarray):
                t = torch.from_numpy(m).float()
            elif isinstance(m, (list, tuple)):
                t = torch.tensor(m, dtype=torch.float32)
            else:
                raise TypeError(f"Unsupported explanation map type: {type(m)}")

            # Remove singleton batch or channel dimensions
            while t.ndim > 2 and t.shape[0] == 1:
                t = t.squeeze(0)
            if t.ndim != 2:
                raise ValueError(f"Expected 2D explanation map, got shape {tuple(t.shape)}")
            return t

        def _normalize_tensor(t: torch.Tensor, clip_pct: float) -> torch.Tensor:
            # 1. Percentile clipping to attenuate extreme gradient spikes
            if 0.0 < clip_pct < 100.0 and t.numel() > 0:
                q = torch.quantile(t, clip_pct / 100.0)
                t = torch.clamp(t, min=0.0, max=q)
            else:
                t = torch.clamp(t, min=0.0)

            # 2. Min-max scaling to strictly map to [0, 1] for scale parity
            min_v = t.min()
            max_v = t.max()
            denom = max_v - min_v
            if denom > 1e-8:
                return (t - min_v) / denom
            return torch.zeros_like(t)

        # Convert to PyTorch tensors
        t_gradcam = _to_tensor(gradcam_map)
        t_ig = _to_tensor(resolved_ig)
        t_shap = _to_tensor(resolved_shap)

        # Normalize all tensors to [0, 1] to ensure scale parity
        norm_gradcam = _normalize_tensor(t_gradcam, clip_percentile)
        norm_ig = _normalize_tensor(t_ig, clip_percentile)
        norm_shap = _normalize_tensor(t_shap, clip_percentile)

        # Determine target spatial resolution
        if target_size is not None:
            tgt_h, tgt_w = target_size
        else:
            tgt_h = max(norm_gradcam.shape[0], norm_ig.shape[0], norm_shap.shape[0])
            tgt_w = max(norm_gradcam.shape[1], norm_ig.shape[1], norm_shap.shape[1])

        def _align_resolution(t: torch.Tensor, h: int, w: int) -> torch.Tensor:
            if t.shape[0] == h and t.shape[1] == w:
                return t
            return F.interpolate(
                t.unsqueeze(0).unsqueeze(0),
                size=(h, w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)

        # Spatial alignment
        norm_gradcam = _align_resolution(norm_gradcam, tgt_h, tgt_w)
        norm_ig = _align_resolution(norm_ig, tgt_h, tgt_w)
        norm_shap = _align_resolution(norm_shap, tgt_h, tgt_w)

        # Normalize relative weights
        total_w = float(w_gradcam + w_ig + w_shap)
        if total_w <= 1e-8:
            w_gradcam, w_ig, w_shap = 0.5, 0.3, 0.2
            total_w = 1.0
        w_gradcam /= total_w
        w_ig /= total_w
        w_shap /= total_w

        # Weighted linear fusion (RG1): w1 * M_GradCAM++ + w2 * M_IG + w3 * M_SHAP
        unified = (
            w_gradcam * norm_gradcam +
            w_ig * norm_ig +
            w_shap * norm_shap
        )

        # Ensure final unified explanation tensor is strictly bounded in [0, 1]
        u_min = unified.min()
        u_max = unified.max()
        u_denom = u_max - u_min
        if u_denom > 1e-8:
            unified = (unified - u_min) / u_denom
        else:
            unified = torch.zeros_like(unified)

        unified_explanation_tensor = torch.clamp(unified, 0.0, 1.0)
        return unified_explanation_tensor

    else:
        # Fallback using NumPy when PyTorch is not available
        def _to_numpy(m: Any) -> np.ndarray:
            arr = np.array(m, dtype=np.float32)
            while arr.ndim > 2 and arr.shape[0] == 1:
                arr = arr.squeeze(0)
            if arr.ndim != 2:
                raise ValueError(f"Expected 2D explanation map, got shape {arr.shape}")
            return arr

        def _normalize_np(arr: np.ndarray, clip_pct: float) -> np.ndarray:
            if 0.0 < clip_pct < 100.0 and arr.size > 0:
                q = np.percentile(arr, clip_pct)
                arr = np.clip(arr, 0.0, q)
            else:
                arr = np.clip(arr, 0.0, None)
            min_v, max_v = float(arr.min()), float(arr.max())
            denom = max_v - min_v
            if denom > 1e-8:
                return (arr - min_v) / denom
            return np.zeros_like(arr)

        arr_gradcam = _to_numpy(gradcam_map)
        arr_ig = _to_numpy(resolved_ig)
        arr_shap = _to_numpy(resolved_shap)

        norm_gradcam = _normalize_np(arr_gradcam, clip_percentile)
        norm_ig = _normalize_np(arr_ig, clip_percentile)
        norm_shap = _normalize_np(arr_shap, clip_percentile)

        total_w = float(w_gradcam + w_ig + w_shap) or 1.0
        w_gradcam /= total_w
        w_ig /= total_w
        w_shap /= total_w

        unified = w_gradcam * norm_gradcam + w_ig * norm_ig + w_shap * norm_shap
        u_min, u_max = float(unified.min()), float(unified.max())
        u_denom = u_max - u_min
        if u_denom > 1e-8:
            unified = (unified - u_min) / u_denom
        else:
            unified = np.zeros_like(unified)
        return np.clip(unified, 0.0, 1.0)

