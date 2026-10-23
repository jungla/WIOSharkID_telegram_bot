"""
Standalone Edge Inference Engine for Elasmobranch Identification.

Uses pure ONNX Runtime (zero PyTorch runtime dependency).
Supports:
1. Single photo or multi-angle photo batch inference.
2. Calibrated confidence score calculation and tiering.
3. Interactive angle recommendation engine when identification is uncertain or ambiguous.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import onnxruntime as ort
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from diagnostic_guide import get_recommendation_for_prediction

class StandaloneElasmobranchClassifier:
    """Standalone on-device classifier using ONNX Runtime."""

    def __init__(
        self,
        onnx_model_path: str = "models/elasmobranch_mobilenet.onnx",
        meta_path: Optional[str] = "models/metadata.json"
    ):
        self.onnx_path = Path(onnx_model_path)
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.onnx_path}")

        # Initialize ONNX Runtime Session (optimized for CPU/NPU)
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(self.onnx_path), opts, providers=["CPUExecutionProvider"])

        # Extract input/output details
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        # Load metadata (from file or ONNX model metadata properties)
        if meta_path and Path(meta_path).exists():
            with open(meta_path, "r") as f:
                self.meta = json.load(f)
            self.species_list = self.meta["species_list"]
            self.img_size = self.meta["input_size"][1]
            self.mean = np.array(self.meta["normalization"]["mean"], dtype=np.float32)
            self.std = np.array(self.meta["normalization"]["std"], dtype=np.float32)
        else:
            # Fallback to defaults
            self.species_list = [
                "Bathytoshia lata", "Galeocerdo cuvier", "Himantura uarnak",
                "Maculabatis ambigua", "Mobula kuhlii", "Neotrygon caeruleopunctata",
                "Pateobatis fai", "Pliotrema annae", "Rhinoptera jayakari",
                "Squalus mahia", "Taeniurops meyeni"
            ]
            self.img_size = 224
            self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def preprocess_image(self, image_input: Union[str, Path, Image.Image]) -> np.ndarray:
        """
        Preprocess single image into normalized NCHW numpy array [1, 3, H, W].
        """
        if isinstance(image_input, (str, Path)):
            image = Image.open(image_input).convert("RGB")
        else:
            image = image_input.convert("RGB")

        # Resize to model input dimension
        resized = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        img_arr = np.array(resized, dtype=np.float32) / 255.0

        # Normalize (img - mean) / std
        norm_img = (img_arr - self.mean) / self.std

        # Transpose HWC -> CHW and add batch dimension -> NCHW
        chw_img = np.transpose(norm_img, (2, 0, 1))
        return np.expand_dims(chw_img, axis=0)

    def predict(
        self,
        image_inputs: Union[str, Path, Image.Image, List[Union[str, Path, Image.Image]]],
        top_k: int = 3,
        high_conf_thresh: float = 0.80,
        mod_conf_thresh: float = 0.60
    ) -> Dict[str, any]:
        """
        Perform species identification for one or multiple photos of an individual.
        Aggregates evidence across multiple angles if a list of photos is provided.
        """
        if not isinstance(image_inputs, list):
            image_inputs = [image_inputs]

        # Preprocess all submitted images into a single batch
        batch_arrays = [self.preprocess_image(img) for img in image_inputs]
        input_tensor = np.concatenate(batch_arrays, axis=0).astype(np.float32)

        # Run ONNX inference
        outputs = self.session.run(None, {self.input_name: input_tensor})
        probabilities_batch = outputs[0]  # shape: (N, num_classes)

        # Multi-Angle Pooling: average probabilities across all submitted photos
        if probabilities_batch.shape[0] > 1:
            mean_probs = np.mean(probabilities_batch, axis=0)
        else:
            mean_probs = probabilities_batch[0]

        # Top-k species ranking
        sorted_indices = np.argsort(mean_probs)[::-1]
        top_results = []
        for idx in sorted_indices[:top_k]:
            sp_name = self.species_list[idx]
            conf = float(mean_probs[idx])
            top_results.append({
                "species": sp_name,
                "confidence": conf,
                "confidence_pct": f"{conf * 100:.1f}%",
                "class_index": int(idx)
            })

        top1 = top_results[0]
        top2 = top_results[1] if len(top_results) > 1 else None

        top1_conf = top1["confidence"]
        top2_conf = top2["confidence"] if top2 else 0.0
        margin = top1_conf - top2_conf

        # Determine confidence tier
        if top1_conf >= high_conf_thresh:
            tier = "High Confidence"
        elif top1_conf >= mod_conf_thresh:
            tier = "Moderate Confidence"
        else:
            tier = "Uncertain"

        # Query diagnostic recommendation engine
        rec = get_recommendation_for_prediction(
            top1_species=top1["species"],
            top1_conf=top1_conf,
            top2_species=top2["species"] if top2 else None,
            top2_conf=top2_conf,
            confidence_tier=tier
        )

        return {
            "top_species": top1["species"],
            "top_confidence": top1_conf,
            "confidence_tier": tier,
            "margin": margin,
            "top_k": top_results,
            "num_photos_analyzed": len(image_inputs),
            "recommendation": rec
        }
