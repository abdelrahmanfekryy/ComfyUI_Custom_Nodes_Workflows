from .nodes import (
    OCRLoadImage,
    OCRLoadDocument,
    OCRGGUFModelLoader,
    OCRProjectorLoader,
    OCRInference,
    OCRInferenceAdvanced,
    OCRDisplayAny,
)

NODE_CLASS_MAPPINGS = {
    "OCRLoadImage": OCRLoadImage,
    "OCRLoadDocument": OCRLoadDocument,
    "OCRGGUFModelLoader": OCRGGUFModelLoader,
    "OCRProjectorLoader": OCRProjectorLoader,
    "OCRInference": OCRInference,
    "OCRInferenceAdvanced": OCRInferenceAdvanced,
    "OCRDisplayAny": OCRDisplayAny,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OCRLoadImage": "📄 Load Image",
    "OCRLoadDocument": "📄 Load Document (PDF / Image)",
    "OCRGGUFModelLoader": "📄 GGUF Model Loader",
    "OCRProjectorLoader": "📄 Projector Loader",
    "OCRInference": "📄 OCR Inference",
    "OCRInferenceAdvanced": "📄 OCR Inference (Advanced)",
    "OCRDisplayAny": "📄 OCR Display Any",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
