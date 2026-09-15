"""
TrustXAI-Med: Direct Model Execution & Diagnostic CLI
Unified Hybrid Explanation Fusion and Uncertainty-Grounded Reliability Framework

Usage:
    python run_model.py
    python run_model.py --image path/to/chest_xray.png
    python run_model.py --serve
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image

# Ensure backend modules are on sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import torch
import torchvision.transforms as transforms

from app.models.real_inference import RealInferenceEngine
from app.xai.real_xai import RealXAIEngine
from app.xai.real_xqi import (
    calculate_faithfulness,
    calculate_localization_iou,
    calculate_robustness,
    calculate_composite_xqi,
    calculate_ers
)


def find_default_image():
    candidates = [
        os.path.join(BASE_DIR, "data", "chexpert", "val", "images", "00000003_002.png"),
        os.path.join(BASE_DIR, "data", "chexpert", "train", "images", "00000001_000.png"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    # Search in data dir
    data_dir = os.path.join(BASE_DIR, "data")
    if os.path.exists(data_dir):
        for root, _, files in os.walk(data_dir):
            for f in files:
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    return os.path.join(root, f)
    return None


def run_pipeline(image_path: str, checkpoint_path: str = None, save_heatmap: str = None):
    print("=" * 78)
    print(" TRUSTXAI-MED: CHEST RADIOGRAPHY MULTI-LABEL INFERENCE & XAI PIPELINE")
    print("=" * 78)

    # 1. Load Model & Weights
    default_ckpt = os.path.join(BASE_DIR, "checkpoints", "nih_chestxray14", "densenet121_nih_chestxray14_best.pth")
    ckpt_to_use = checkpoint_path if checkpoint_path else (default_ckpt if os.path.exists(default_ckpt) else "")

    print(f"\n[1/5] Loading Model Architecture: DenseNet-121")
    if ckpt_to_use and os.path.exists(ckpt_to_use):
        print(f"      Checkpoint: {os.path.relpath(ckpt_to_use, BASE_DIR)}")
    else:
        print("      Checkpoint: NIH Checkpoint not specified or missing; using standard initialization.")

    loaded = RealInferenceEngine.load_model(checkpoint_path=ckpt_to_use)
    model = loaded["model"]
    classes = loaded["classes"]
    device = loaded["device"]
    print(f"      Target Classes ({len(classes)}): {', '.join(classes[:5])}...")
    print(f"      Execution Device: {device}")

    # 2. Load Input Image
    if not image_path or not os.path.exists(image_path):
        found = find_default_image()
        if not found:
            print("\n[ERROR] No image provided and no sample images found in data/ directory.")
            sys.exit(1)
        image_path = found

    print(f"\n[2/5] Preprocessing Input Image:")
    print(f"      Image Path: {os.path.relpath(image_path, BASE_DIR) if os.path.isabs(image_path) else image_path}")
    raw_img = Image.open(image_path).convert("RGB")
    print(f"      Original Dimensions: {raw_img.size[0]} x {raw_img.size[1]} px")

    # 3. Deterministic + Monte Carlo Inference
    print(f"\n[3/5] Running Deterministic Forward Pass & MC Dropout Uncertainty (T=20)...")
    inf_res = RealInferenceEngine.run_inference(raw_img)

    print("\n      " + "-" * 60)
    print(f"      PREDICTION SUMMARY:")
    print(f"      Primary Pathology:  {inf_res.predicted_label.upper()}")
    print(f"      Confidence / Prob:  {inf_res.confidence * 100:.2f}%")
    print(f"      Epistemic Entropy:  {inf_res.entropy:.4f}")
    print(f"      Predictive Var:     {inf_res.mc_variance:.6f} (sigma^2_MC)")
    print(f"      Uncertainty Score:  {inf_res.uncertainty_score:.4f} ({inf_res.uncertainty_level})")
    print("      " + "-" * 60)

    print("\n      Pathology Probabilities (Top Findings):")
    sorted_probs = sorted(inf_res.probabilities.items(), key=lambda x: x[1], reverse=True)
    for pathology, prob in sorted_probs:
        bar = "#" * int(prob * 30)
        marker = " <== TOP" if pathology == inf_res.predicted_label else ""
        print(f"        {pathology:<20}: {prob * 100:5.2f}% | {bar:<30}{marker}")

    # 4. Tri-Modal Saliency & Fusion
    print(f"\n[4/5] Computing Tri-Modal Explainability Maps (Grad-CAM++, IG, Superpixel SHAP)...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(raw_img).unsqueeze(0).to(device)
    top_idx = classes.index(inf_res.predicted_label) if inf_res.predicted_label in classes else 0

    gcam = RealXAIEngine.generate_gradcam_plus_plus(model, img_tensor, top_idx)
    ig = RealXAIEngine.generate_integrated_gradients(model, img_tensor, top_idx, steps=25)
    shap = RealXAIEngine.generate_superpixel_shap(model, img_tensor, top_idx)

    fused = RealXAIEngine.hybrid_explanation_fusion(
        gradcam_map=gcam,
        ig_map=ig,
        shap_map=shap,
        alpha=0.4,
        beta=0.3,
        gamma=0.3
    )
    print(f"      Grad-CAM++ Range: [{np.min(gcam):.4f}, {np.max(gcam):.4f}]")
    print(f"      Integrated Grad Range: [{np.min(ig):.4f}, {np.max(ig):.4f}]")
    fused_np = fused.detach().cpu().numpy() if hasattr(fused, "detach") else np.array(fused)
    print(f"      Hybrid Fused Saliency: [{float(np.min(fused_np)):.4f}, {float(np.max(fused_np)):.4f}], Shape: {fused_np.shape}")

    # 5. XQI Evaluation & Clinical Reliability Coupling
    print(f"\n[5/5] Evaluating Explanation Quality (XQI) & Clinical Reliability (ERS)...")
    faith = calculate_faithfulness(model, img_tensor, fused_np, target_class_idx=top_idx, mask_ratio=0.20)
    robustness = calculate_robustness(
        fused_explanation=fused_np,
        model=model,
        image_tensor=img_tensor,
        explainer_func=lambda m, x: RealXAIEngine.generate_gradcam_plus_plus(m, x, top_idx, grid_size=32),
        noise_std=0.05
    )

    xqi_composite, weights_used = calculate_composite_xqi(
        faithfulness=faith,
        localization=None,
        robustness=robustness
    )
    ers_score, clinical_flag, is_verified = calculate_ers(
        xqi_score=xqi_composite,
        composite_uncertainty=inf_res.uncertainty_score,
        mc_variance=inf_res.mc_variance
    )

    print("\n" + "=" * 78)
    print(" CLINICAL RELIABILITY EVALUATION REPORT (RG1 - RG4)")
    print("=" * 78)
    print(f"  * Faithfulness (Pixel-Deletion Drop Delta P): {faith:.2f} / 100")
    print(f"  * Saliency Robustness (sigma=0.05 Noise):      {robustness:.2f} / 100")
    print(f"  * Composite Explanation Quality Index (XQI):  {xqi_composite:.2f} / 100")
    print(f"  * Epistemic Uncertainty (U):                  {inf_res.uncertainty_score:.4f}")
    print(f"  * MC Predictive Variance (sigma^2_MC):        {inf_res.mc_variance:.6f}")
    print(f"  * Explanation Reliability Score (ERS):        {ers_score:.2f} / 100")
    print(f"  * Clinical Safety Gating Flag:                [{clinical_flag}]")
    print("=" * 78)

    # Optional heatmap save
    if save_heatmap:
        try:
            # Side-by-side visualization: Input Radiograph + Fused Heatmap Overlay using PIL
            base_224 = raw_img.resize((224, 224)).convert("RGB")
            sal_norm = np.clip(fused_np, 0.0, 1.0)
            sal_img = Image.fromarray((sal_norm * 255).astype(np.uint8), mode='L').resize((224, 224), Image.Resampling.BILINEAR)
            sal_arr = np.array(sal_img, dtype=np.float32) / 255.0

            # Jet-inspired colormap (R, G, B)
            color_map = np.zeros((224, 224, 3), dtype=np.uint8)
            color_map[:, :, 0] = (np.clip(2.0 * sal_arr - 0.5, 0.0, 1.0) * 255).astype(np.uint8)
            color_map[:, :, 1] = (np.clip(1.0 - 2.0 * np.abs(sal_arr - 0.5), 0.0, 1.0) * 255).astype(np.uint8)
            color_map[:, :, 2] = (np.clip(1.0 - 2.0 * sal_arr, 0.0, 1.0) * 255).astype(np.uint8)

            heat_img = Image.fromarray(color_map, mode='RGB')
            blended = Image.blend(base_224, heat_img, alpha=0.45)

            # Combined canvas (Side-by-side)
            canvas = Image.new("RGB", (468, 234), (25, 25, 30))
            canvas.paste(base_224, (5, 5))
            canvas.paste(blended, (239, 5))
            canvas.save(save_heatmap)
            print(f"\n[INFO] Fused explanation visualization saved to: {save_heatmap}")
        except Exception as e:
            print(f"\n[WARN] Could not save heatmap visualization: {e}")

    return {
        "prediction": inf_res.predicted_label,
        "confidence": inf_res.confidence,
        "ers": ers_score,
        "clinical_flag": clinical_flag
    }


def launch_server(port: int = 8008):
    import uvicorn
    print(f"\n[INFO] Launching TrustXAI-Med API server at http://127.0.0.1:{port}...")
    print(f"[INFO] Interactive Swagger UI available at: http://127.0.0.1:{port}/docs\n")
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=port, reload=False)


def main():
    parser = argparse.ArgumentParser(description="TrustXAI-Med Model Runner")
    parser.add_argument("--image", "-i", type=str, default=None, help="Path to input chest X-ray image")
    parser.add_argument("--checkpoint", "-c", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--save-heatmap", "-s", type=str, default="explanation_output.png", help="Path to save fused explanation image")
    parser.add_argument("--serve", action="store_true", help="Start the FastAPI backend server on port 8008")
    parser.add_argument("--port", type=int, default=8008, help="API server port (used with --serve)")
    args = parser.parse_args()

    if args.serve:
        launch_server(args.port)
    else:
        run_pipeline(args.image, args.checkpoint, args.save_heatmap)


if __name__ == "__main__":
    main()
