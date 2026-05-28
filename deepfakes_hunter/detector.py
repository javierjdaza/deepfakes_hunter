import pathlib
import numpy as np
import cv2
import onnxruntime as ort
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

_MODEL_URL = "https://drive.google.com/uc?id=1xPD_velem0PIaHHi8ysgwxkA9FA7tq5n"
_DEFAULT_MODEL_PATH = pathlib.Path.home() / ".deepfakes_hunter" / "deepfake_detector.onnx"


def _ensure_model(dest: pathlib.Path) -> None:
    if dest.exists():
        return
    try:
        import gdown
    except ImportError as e:
        raise ImportError(
            "gdown is required for automatic model download: pip install gdown"
        ) from e
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading deepfake detector model to {dest} ...")
    gdown.download(_MODEL_URL, str(dest), quiet=False)


class DeepfakeDetector:
    IMG_SIZE = 224
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    def __init__(self, onnx_path: str = None, threshold: float = 0.98, use_gpu: bool = False):
        if onnx_path is None:
            _ensure_model(_DEFAULT_MODEL_PATH)
            onnx_path = str(_DEFAULT_MODEL_PATH)

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.threshold = threshold

        self.transform = A.Compose([
            A.Resize(self.IMG_SIZE, self.IMG_SIZE, interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
            ToTensorV2(),
        ])

    def predict(self, img_pillow, threshold: float = None):
        thr = threshold if threshold is not None else self.threshold

        img_array = np.array(img_pillow.convert('RGB'))
        tensor = self.transform(image=img_array)['image'].unsqueeze(0).numpy()

        logit = self.session.run(['logit'], {'image': tensor})[0]
        score = float(1 / (1 + np.exp(-logit[0])))
        pred = int(score >= thr)

        return {
            'score': score,
            'pred': pred,
            'label': 'deepfake' if pred == 1 else 'real',
            'threshold': thr,
        }
