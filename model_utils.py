"""
model_utils.py
Model loading, prediction, and confidence thresholding utilities
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import io
from typing import Dict, List, Tuple, Union, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_pil_resample():
    """Pillow 10+ moved resampling enums under Image.Resampling."""
    try:
        return Image.Resampling.BILINEAR
    except Exception:
        return Image.BILINEAR


class GarbageClassifier:
    """
    Wrapper class for garbage classification model
    """

    def __init__(
        self,
        model_path: str = "models/best_model.keras",
        img_size: Tuple[int, int] = (300, 300),
        confidence_threshold: float = 0.70,
        out_of_scope_threshold: float = 0.50
    ):
        """
        Initialize the garbage classifier

        Args:
            model_path: Path to the trained Keras model
            img_size: Target image size for model input (will auto-sync to model if different)
            confidence_threshold: Minimum confidence for single prediction
            out_of_scope_threshold: Maximum confidence to consider "out of scope"
        """
        self.model_path = model_path
        self.img_size = img_size  # will be validated/overridden to match model input
        self.confidence_threshold = confidence_threshold
        self.out_of_scope_threshold = out_of_scope_threshold

        # Class names (must match training order)
        self.class_names = [
            'battery', 'biological', 'brown-glass', 'cardboard',
            'clothes', 'green-glass', 'metal', 'paper',
            'plastic', 'shoes', 'trash', 'white-glass'
        ]

        # Load model
        self.model = None
        self.load_model()

    def _resolve_model_input_shape(self) -> Tuple[int, int, int]:
        """
        Returns (height, width, channels) expected by the model.
        Handles common cases: single input tensor (None, H, W, C).
        """
        input_shape = self.model.input_shape

        # Some models have multiple inputs -> input_shape is a list/tuple of shapes
        if isinstance(input_shape, (list, tuple)) and len(input_shape) > 0 and isinstance(input_shape[0], (list, tuple)):
            # Pick first input by default
            input_shape = input_shape[0]

        # Expected channels_last: (None, H, W, C)
        if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 4:
            raise ValueError(f"Unsupported model input_shape: {self.model.input_shape}. Expected (None, H, W, C).")

        _, h, w, c = input_shape

        if h is None or w is None:
            raise ValueError(
                f"Model has dynamic spatial input shape {self.model.input_shape}. "
                "Please set img_size explicitly to a fixed size your model supports."
            )

        if c is None:
            # Most likely fine, but we can't validate channels count
            c = 3

        return int(h), int(w), int(c)

    def load_model(self):
        """Load the trained model and auto-sync preprocessing size to model input."""
        try:
            logger.info(f"Loading model from {self.model_path}")
            self.model = tf.keras.models.load_model(self.model_path)
            logger.info("✅ Model loaded successfully")
            logger.info(f"   Input shape: {self.model.input_shape}")
            logger.info(f"   Output shape: {self.model.output_shape}")

            h, w, c = self._resolve_model_input_shape()
            if c != 3:
                raise ValueError(f"Model expects {c} channels, but only RGB(3) is supported.")

            expected = (w, h)  # PIL expects (width, height)
            if tuple(self.img_size) != expected:
                logger.warning(
                    f"⚠️ img_size {self.img_size} does not match model input {(h, w, c)}. "
                    f"Overriding img_size to {expected}."
                )
                self.img_size = expected

            logger.info(f"   Using img_size: {self.img_size}")

        except Exception as e:
            logger.error(f"❌ Failed to load model: {str(e)}")
            raise

    def preprocess_image(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """
        Preprocess image for model input:
          - convert to RGB
          - resize to model expected size
          - convert to float32
          - add batch dimension
        """
        # Convert to PIL Image if numpy array
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize to model's required input size
        resample = _get_pil_resample()
        image = image.resize(self.img_size, resample=resample)

        # Convert to array and add batch dimension
        img_array = np.array(image, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)

        # If your model was trained with 0-1 normalization and does NOT include Rescaling in the model,
        # uncomment this line:
        # img_array = img_array / 255.0

        return img_array

    def predict(self, image: Union[Image.Image, np.ndarray, bytes]) -> Dict[str, Union[str, float, List, Dict]]:
        """
        Make prediction on a single image

        Args:
            image: PIL Image, numpy array, or image bytes

        Returns:
            Dictionary containing prediction results
        """
        # Handle bytes input
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))

        # Preprocess
        img_array = self.preprocess_image(image)

        # Predict
        preds = self.model.predict(img_array, verbose=0)
        # Handle models that return a list (rare but possible)
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        predictions = preds[0]

        # Get top prediction
        top_idx = int(np.argmax(predictions))
        top_class = self.class_names[top_idx]
        top_confidence = float(predictions[top_idx])

        # Get top 3 predictions
        top_3_indices = np.argsort(predictions)[-3:][::-1]
        top_3 = [
            {
                'class': self.class_names[int(idx)],
                'confidence': float(predictions[int(idx)]),
                'rank': rank + 1
            }
            for rank, idx in enumerate(top_3_indices)
        ]

        # Determine prediction status
        if top_confidence < self.out_of_scope_threshold:
            status = 'out_of_scope'
        elif top_confidence >= self.confidence_threshold:
            status = 'confident'
        else:
            status = 'uncertain'

        return {
            'predicted_class': top_class,
            'confidence': top_confidence,
            'status': status,
            'top_3': top_3,
            'all_probabilities': {
                self.class_names[i]: float(predictions[i])
                for i in range(len(self.class_names))
            }
        }

    def predict_batch(self, images: List[Union[Image.Image, np.ndarray, bytes]]) -> List[Dict[str, Any]]:
        """Make predictions on multiple images."""
        return [self.predict(img) for img in images]

    def predict_with_tta(
        self,
        image: Union[Image.Image, np.ndarray],
        num_augmentations: int = 5
    ) -> Dict[str, Union[str, float, List, Dict]]:
        """
        Make prediction with Test-Time Augmentation (TTA)
        """
        # Original prediction
        img_array = self.preprocess_image(image)
        preds0 = self.model.predict(img_array, verbose=0)
        if isinstance(preds0, (list, tuple)):
            preds0 = preds0[0]
        predictions_list = [preds0[0]]

        # Augmented predictions
        for _ in range(num_augmentations - 1):
            aug_image = self._augment_image(image)
            aug_array = self.preprocess_image(aug_image)
            pr = self.model.predict(aug_array, verbose=0)
            if isinstance(pr, (list, tuple)):
                pr = pr[0]
            predictions_list.append(pr[0])

        # Average predictions
        avg_predictions = np.mean(predictions_list, axis=0)

        # Get results
        top_idx = int(np.argmax(avg_predictions))
        top_class = self.class_names[top_idx]
        top_confidence = float(avg_predictions[top_idx])

        top_3_indices = np.argsort(avg_predictions)[-3:][::-1]
        top_3 = [
            {
                'class': self.class_names[int(idx)],
                'confidence': float(avg_predictions[int(idx)]),
                'rank': rank + 1
            }
            for rank, idx in enumerate(top_3_indices)
        ]

        if top_confidence < self.out_of_scope_threshold:
            status = 'out_of_scope'
        elif top_confidence >= self.confidence_threshold:
            status = 'confident'
        else:
            status = 'uncertain'

        return {
            'predicted_class': top_class,
            'confidence': top_confidence,
            'status': status,
            'top_3': top_3,
            'all_probabilities': {
                self.class_names[i]: float(avg_predictions[i])
                for i in range(len(self.class_names))
            },
            'tta_applied': True
        }

    def _augment_image(self, image: Image.Image) -> Image.Image:
        """Apply random augmentation to image."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Random horizontal flip
        if np.random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # Random rotation (-15 to 15 degrees)
        angle = np.random.uniform(-15, 15)
        image = image.rotate(angle, fillcolor=(255, 255, 255))

        # Random brightness adjustment
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Brightness(image)
        factor = np.random.uniform(0.8, 1.2)
        image = enhancer.enhance(factor)

        return image

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self.model is None:
            return {'error': 'Model not loaded'}

        total_params = self.model.count_params()
        return {
            'model_path': self.model_path,
            'input_shape': self.model.input_shape,
            'output_shape': self.model.output_shape,
            'total_parameters': total_params,
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'confidence_threshold': self.confidence_threshold,
            'out_of_scope_threshold': self.out_of_scope_threshold,
            'img_size_used': self.img_size
        }

    def validate_prediction(self, predicted_class: str, actual_class: str) -> bool:
        """Validate if prediction matches actual class."""
        return predicted_class.lower() == actual_class.lower()


class PredictionLogger:
    """Logger for tracking predictions and collecting feedback."""

    def __init__(self, log_file: str = "feedback/predictions.log"):
        self.log_file = log_file
        self._ensure_log_file()

    def _ensure_log_file(self):
        import os
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', encoding="utf-8") as f:
                f.write("timestamp,predicted_class,confidence,status,correct,user_feedback\n")

    def log_prediction(
        self,
        predicted_class: str,
        confidence: float,
        status: str,
        correct: bool = None,
        user_feedback: str = None
    ):
        import datetime
        timestamp = datetime.datetime.now().isoformat()

        with open(self.log_file, 'a', encoding="utf-8") as f:
            f.write(f"{timestamp},{predicted_class},{confidence:.4f},{status},")
            f.write(f"{correct if correct is not None else 'NA'},")
            f.write(f"{user_feedback if user_feedback else 'NA'}\n")


if __name__ == "__main__":
    print("Initializing classifier...")
    classifier = GarbageClassifier(
        model_path="models/garbage_classifier_transfer_learning_model_b.keras",
        confidence_threshold=0.70,
        out_of_scope_threshold=0.50
    )

    print("\nModel Info:")
    info = classifier.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")

    print("\nTesting with dummy image...")
    dummy_image = Image.new('RGB', (300, 300), color='red')

    result = classifier.predict(dummy_image)
    print(f"\nPrediction Result:")
    print(f"  Class: {result['predicted_class']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  Status: {result['status']}")
    print(f"  Top 3:")
    for item in result['top_3']:
        print(f"    {item['rank']}. {item['class']}: {item['confidence']:.2%}")
