import time
import torch
from .metrics import classification_metrics


def evaluate(model, loader, checkpoint, device):
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    actual, predicted, scores = [], [], []
    inference_seconds = 0.0
    with torch.no_grad():
        for images, labels in loader:
            if str(device).startswith("cuda"):
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            logits = model(images.to(device))
            if str(device).startswith("cuda"):
                torch.cuda.synchronize(device)
            inference_seconds += time.perf_counter() - started
            actual.extend(labels.tolist())
            predicted.extend(logits.argmax(1).cpu().tolist())
            scores.extend(logits.softmax(1).cpu().tolist())
    result = classification_metrics(actual, predicted, scores)
    result["inference_seconds"] = inference_seconds
    result["latency_ms_per_image"] = inference_seconds * 1000.0 / max(1, len(actual))
    result["total_parameters"] = sum(parameter.numel() for parameter in model.parameters())
    result["trainable_parameters"] = sum(parameter.numel() for parameter in model.parameters()
                                          if parameter.requires_grad)
    return result
