import os
import sys
import subprocess

for pkg in ["numpy", "cv2", "onnxruntime"]:
    try:
        __import__(pkg)
    except ImportError:
        pip_name = "opencv-contrib-python" if pkg == "cv2" else pkg
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])

import cv2
import numpy as np
import onnxruntime as ort
import time
import urllib.request
from collections import Counter

# Enter the path to your video file below:
VIDEO_PATH = r"rej_happy.mp4"


class EmotiEffLibModel:
    def __init__(self):
        self.img_size = 224
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.classes = {
            0: "Anger", 1: "Contempt", 2: "Disgust", 3: "Fear",
            4: "Happiness", 5: "Neutral", 6: "Sadness", 7: "Surprise"
        }

        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "enet_b0_8_va_mtl.onnx")
        if not os.path.exists(model_path):
            cache_path = os.path.join(os.path.expanduser("~"), ".emotiefflib", "enet_b0_8_va_mtl.onnx")
            if os.path.exists(cache_path):
                model_path = cache_path
            else:
                url = "https://github.com/sb-ai-lab/EmotiEffLib/blob/main/models/affectnet_emotions/onnx/enet_b0_8_va_mtl.onnx?raw=true"
                print("Downloading pretrained EfficientNet-B0 model weights...")
                urllib.request.urlretrieve(url, model_path)

        cpu_cores = os.cpu_count() or 4
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = cpu_cores
        sess_opts.inter_op_num_threads = cpu_cores
        sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self.session = ort.InferenceSession(model_path, sess_opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, face_rgb):
        x = cv2.resize(face_rgb, (self.img_size, self.img_size)) / 255.0
        x = (x - self.mean) / self.std
        tensor = x.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

        raw_output = self.session.run(None, {self.input_name: tensor})[0]
        logits = raw_output[0][:8]
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        dominant = self.classes[int(np.argmax(probs))]
        return dominant, probs


def locate_video(path):
    if os.path.isabs(path) and os.path.exists(path):
        return path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    for candidate in [
        os.path.join(cwd, path),
        os.path.join(script_dir, path),
        os.path.join(script_dir, "..", path),
        os.path.join(cwd, "send_emo", path),
        os.path.join(script_dir, "..", "emo_model", path),
        os.path.join(cwd, "emo_model", path),
    ]:
        norm = os.path.abspath(candidate)
        if os.path.exists(norm):
            return norm
    return path


def run():
    target_video = locate_video(VIDEO_PATH)
    if not os.path.exists(target_video):
        print("Error: Video file not found at:", VIDEO_PATH)
        return

    t_start = time.perf_counter()
    t_prep_start = time.perf_counter()

    cap = cv2.VideoCapture(target_video)
    if not cap.isOpened():
        print("Error: Could not open video:", target_video)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = 30.0 if fps <= 0 else fps
    total_frames = 300 if total_frames <= 0 else total_frames
    duration_sec = total_frames / fps

    cpu_cores = os.cpu_count() or 4
    print("=" * 60)
    print("       EMOTIEFFLIB EFFICIENTNET-B0 EMOTION RECOGNITION")
    print("=" * 60)
    print(f"Video File       : {os.path.basename(target_video)}")
    print(f"Duration         : {duration_sec:.2f} seconds")
    print(f"Resolution       : {width}x{height} @ {fps:.1f} FPS")
    print(f"CPU Threads Used : {cpu_cores} cores")

    frame_step = max(1, int(round(fps / 2.0)))
    sampled_indices = set(range(0, total_frames, frame_step))

    sampled_frames = []
    curr_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if curr_idx in sampled_indices:
            sampled_frames.append(frame)
        curr_idx += 1
    cap.release()

    t_prep = time.perf_counter() - t_prep_start
    print(f"Sampled Frames   : {len(sampled_frames)}")
    print("-" * 60)

    t_load_start = time.perf_counter()
    model = EmotiEffLibModel()
    t_load = time.perf_counter() - t_load_start

    dummy = np.zeros((model.img_size, model.img_size, 3), dtype=np.uint8)
    _ = model.predict(dummy)

    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(cascade_path)

    frame_probs = []
    frame_dominants = []
    t_det = 0.0
    t_inf = 0.0
    failures = 0

    for frame in sampled_frames:
        t_d_start = time.perf_counter()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
        t_det += time.perf_counter() - t_d_start

        if len(faces) == 0:
            failures += 1
            continue

        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        pad_w, pad_h = int(w * 0.15), int(h * 0.15)
        x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
        x2, y2 = min(width, x + w + pad_w), min(height, y + h + pad_h)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            failures += 1
            continue

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        t_i_start = time.perf_counter()
        dom, probs = model.predict(crop_rgb)
        t_inf += time.perf_counter() - t_i_start

        frame_probs.append(probs)
        frame_dominants.append(dom)

    total_time = time.perf_counter() - t_start
    valid_count = len(frame_probs)

    if valid_count == 0:
        print("No faces detected in the video.")
        print(f"Detection Failures : {failures}")
        print(f"Total Time         : {total_time:.4f} seconds")
        return

    mean_probs = np.mean(frame_probs, axis=0)
    top_emotion = model.classes[int(np.argmax(mean_probs))]
    mode_emotion, mode_count = Counter(frame_dominants).most_common(1)[0]

    print("\nEMOTION PROBABILITIES (Aggregated Across Frames):")
    print("-" * 60)
    for i in range(8):
        name = model.classes[i]
        pct = mean_probs[i] * 100.0
        bar = "█" * int(round(pct / 4.0))
        print(f"  {name:<12}: {pct:6.2f}%  {bar}")

    print("-" * 60)
    print(f"Dominant Emotion       : {top_emotion}")
    print(f"Most Frequent Emotion  : {mode_emotion} ({mode_count}/{valid_count} frames)")
    print(f"Processed Frames       : {valid_count}/{len(sampled_frames)} (Failures: {failures})")

    avg_frame_ms = (t_inf / valid_count) * 1000.0
    print("-" * 60)
    print("PROCESSING TIME BREAKDOWN:")
    print("-" * 60)
    print(f"  Model Loading Time         : {t_load:.4f} seconds")
    print(f"  Video Preprocessing Time   : {t_prep:.4f} seconds")
    print(f"  Face Detection & Crop Time : {t_det:.4f} seconds")
    print(f"  Model Inference Time       : {t_inf:.4f} seconds")
    print(f"  Avg Inference Time / Frame : {avg_frame_ms:.2f} ms")
    print(f"  Total Processing Time      : {total_time:.4f} seconds")
    print("=" * 60)


if __name__ == "__main__":
    run()
