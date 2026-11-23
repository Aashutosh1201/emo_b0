# emo_b0

Fast, standalone facial emotion recognition using pretrained **EfficientNet-B0** powered by ONNX Runtime. It runs entirely on CPU using your system's full processor capacity, with no virtual environment required.

The pretrained model file (`enet_b0_8_va_mtl.onnx`, 16 MB) is bundled directly in this repository.

---

## Instructions for Running

### Step 1: Clone or Download
Download or clone this repository to your computer and open your terminal in this folder:

```bash
git clone https://github.com/Aashutosh1201/emo_b0.git
cd emo_b0
```

---

### Step 2: Set Your Video Path
Open `run_emotion.py` in any text editor and update line 19 with your video file name or path:

```python
# Enter the path to your video file below:
VIDEO_PATH = r"my_video.mp4"
```

*(You can simply copy your video file directly into this folder and write its name, or provide an absolute path).*

---

### Step 3: Run the Script

Simply run:

```bash
python run_emotion.py
```

*(On the very first run, if `numpy`, `opencv-contrib-python`, or `onnxruntime` are missing, the script will automatically install them for you).*

---

## Output

The script will:
1. Detect and allocate all available CPU cores on your machine.
2. Load the bundled `enet_b0_8_va_mtl.onnx` weights directly from this folder.
3. Detect faces across video frames and predict emotion probabilities (`Anger`, `Contempt`, `Disgust`, `Fear`, `Happiness`, `Neutral`, `Sadness`, `Surprise`).
4. Output the dominant emotion, most frequent frame emotion, and detailed processing times.
