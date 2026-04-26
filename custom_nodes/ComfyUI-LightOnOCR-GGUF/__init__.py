from .nodes import (
    OCRLoadImage,
    OCRGGUFModelLoader,
    OCRProjectorLoader,
    OCRInference,
    OCRInferenceAdvanced,
    OCRShowText,
)

NODE_CLASS_MAPPINGS = {
    "OCRLoadImage": OCRLoadImage,
    "OCRGGUFModelLoader": OCRGGUFModelLoader,
    "OCRProjectorLoader": OCRProjectorLoader,
    "OCRInference": OCRInference,
    "OCRInferenceAdvanced": OCRInferenceAdvanced,
    "OCRShowText": OCRShowText,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OCRLoadImage": "📄 Load Image",
    "OCRGGUFModelLoader": "📄 GGUF Model Loader",
    "OCRProjectorLoader": "📄 Projector Loader",
    "OCRInference": "📄 OCR Inference",
    "OCRInferenceAdvanced": "📄 OCR Inference (Advanced)",
    "OCRShowText": "📄 Show Text",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
