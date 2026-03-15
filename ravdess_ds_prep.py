import os
import re
import cv2
import math
import json
import shutil
import random
import subprocess
from tqdm import tqdm
import pandas as pd
import numpy as np


EMOTION_MAP = {
    3: "happy",
    4: "sad",
    5: "anger",
    6: "fear",
    7: "disgust"
}

def parse_ravdess_filename(path: str):
    """Returns dict with actor_id, gender, emotion label, etc. or None if not RAVDESS format."""
    base = os.path.basename(path)
    m = re.match(r"(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})", base)
    if not m:
        return None
    parts = list(map(int, m.groups()))
    emotion_code = parts[2]
    actor_id = parts[6]
    emotion = EMOTION_MAP.get(emotion_code, None)
    if emotion is None:
        return None
    # RAVDESS: odd actor = male, even actor = female
    gender = "male" if (actor_id % 2 == 1) else "female"
    return {
        "actor_id": actor_id,
        "gender": gender,
        "emotion": emotion,
        "emotion_code": emotion_code
    }


def create_info_dataframe(path = os.path.join('datasets', 'ravdess')):
    df = pd.DataFrame(columns=['filepath','label','speaker','gender'])
    for actor in os.listdir(path):
        for file in os.listdir(os.path.join(path, actor)):
            info = parse_ravdess_filename(os.path.join(path, actor, file))
            if info is None:
                continue
            df.loc[len(df)] = [os.path.join(path, actor, file), info['emotion'], info['actor_id'], info['gender']]

    return df

df = create_info_dataframe()

# ---------------------------
# Step 4 (Modified): Fixed actor-balanced split
# Train: 16 actors (8M, 8F)
# Val:    2 actors (1M, 1F)
# Test:   6 actors (3M, 3F)
# ---------------------------


rng = np.random.default_rng(42)

# Unique actors and their genders (gender is consistent per actor in RAVDESS)
actor_gender = (
    df[["speaker", "gender"]]
    .drop_duplicates()
    .sort_values("speaker")
)

male_actors = actor_gender[actor_gender["gender"] == "male"]["speaker"].tolist()
female_actors = actor_gender[actor_gender["gender"] == "female"]["speaker"].tolist()

assert len(male_actors) == 12 and len(female_actors) == 12, "Expected 12 male and 12 female actors."

# Shuffle actors reproducibly
rng.shuffle(male_actors)
rng.shuffle(female_actors)

# Allocate actors by counts
train_actors = set(male_actors[:8] + female_actors[:8])
val_actors   = set(male_actors[8:9] + female_actors[8:9])      # 1 male, 1 female
test_actors  = set(male_actors[9:12] + female_actors[9:12])    # 3 male, 3 female

# Sanity: ensure partitions are disjoint and complete
assert len(train_actors) == 16 and len(val_actors) == 2 and len(test_actors) == 6
assert len(train_actors & val_actors) == 0
assert len(train_actors & test_actors) == 0
assert len(val_actors & test_actors) == 0
assert len(train_actors | val_actors | test_actors) == 24

# Apply split to dataframe
df["split"] = "train"
df.loc[df["speaker"].isin(val_actors), "split"] = "val"
df.loc[df["speaker"].isin(test_actors), "split"] = "test"

# Save updated CSV (same as before)
# csv_out = os.path.join('datasets', 'ravdess', "updated_labels.csv")
# df.to_csv(csv_out, index=False)
# print("Saved with split:", csv_out)

# Quick checks
print(df["split"].value_counts())
print("\nActors per split:")
print("Train actors:", sorted(list(train_actors)))
print("Val actors:", sorted(list(val_actors)))
print("Test actors:", sorted(list(test_actors)))

print("\nGender distribution by split:")
print(df.groupby(["split", "gender"]).size().unstack(fill_value=0))

print("\nClass distribution by split:")
print(df.groupby(["split", "label"]).size().unstack(fill_value=0))







'''
from sklearn.model_selection import GroupShuffleSplit

# ---------------------------
# Config
# ---------------------------
RAVDESS_VIDEO_ROOT = "datasets\\ravdess"  # change
OUT_ROOT = "datasets\\ravdess_prepared"         # change

FRAMES_PER_CLIP = 16
FRAME_SIZE = 96            # IMPORTANT: match AffectNet training size (you said 96x96)
AUDIO_SR = 16000
AUDIO_CHANNELS = 1

# If you want exactly 3 seconds crop/pad later in dataloader, keep as-is.
# Here we only extract whole audio track; trimming can be done in dataset class.
EXTRACT_AUDIO = True

# Create output dirs
frames_root = os.path.join(OUT_ROOT, "frames")
audio_root = os.path.join(OUT_ROOT, "audio")
os.makedirs(frames_root, exist_ok=True)
os.makedirs(audio_root, exist_ok=True)


# ---------------------------
# RAVDESS filename parsing
# Example filename:
# 03-01-05-01-02-01-12.mp4
# Fields: modality-channel-emotion-intensity-statement-repetition-actor
# Emotion code mapping (RAVDESS):
# 01 neutral, 02 calm, 03 happy, 04 sad, 05 angry, 06 fearful, 07 disgust, 08 surprised
# ---------------------------

EMOTION_MAP = {
    3: "happy",
    4: "sad",
    5: "anger",
    6: "fear",
    7: "disgust"
}

def parse_ravdess_filename(path: str):
    """Returns dict with actor_id, gender, emotion label, etc. or None if not RAVDESS format."""
    base = os.path.basename(path)
    m = re.match(r"(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})", base)
    if not m:
        return None
    parts = list(map(int, m.groups()))
    emotion_code = parts[2]
    actor_id = parts[6]
    emotion = EMOTION_MAP.get(emotion_code, None)
    if emotion is None:
        return None
    # RAVDESS: odd actor = male, even actor = female
    gender = "male" if (actor_id % 2 == 1) else "female"
    return {
        "actor_id": actor_id,
        "gender": gender,
        "emotion": emotion,
        "emotion_code": emotion_code
    }


def center_crop_square(img):
    """Crop the center square from an image."""
    h, w = img.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return img[y0:y0+side, x0:x0+side]


def preprocess_frame_bgr(frame_bgr, out_size=96):
    """
    RAVDESS frames are 1280x720. We:
    - center-crop to square (keeps face centered)
    - resize to out_size x out_size
    """
    sq = center_crop_square(frame_bgr)
    resized = cv2.resize(sq, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return resized


def extract_uniform_frames(video_path, out_dir, k=16, out_size=96):
    """
    Extract k frames uniformly using OpenCV (no ffmpeg).
    Saves as JPGs: frame_000.jpg ... frame_{k-1}.jpg
    Returns True/False.
    """
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        return False

    # Uniform indices across the full clip
    idxs = np.linspace(0, frame_count - 1, k).round().astype(int)

    saved = 0
    idx_set = set(idxs.tolist())
    wanted = sorted(list(idx_set))

    # Efficient read: iterate frames and save only needed indices
    curr = 0
    wi = 0
    target = wanted[wi]

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if curr == target:
            frame = preprocess_frame_bgr(frame, out_size=out_size)
            out_path = os.path.join(out_dir, f"frame_{saved:03d}.jpg")
            cv2.imwrite(out_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            saved += 1
            wi += 1
            if wi >= len(wanted):
                break
            target = wanted[wi]
        curr += 1

    cap.release()

    # If duplicates collapsed (rare), ensure exactly k images by repeating last
    # or by relaxing idx_set logic. For simplicity, pad by copying last.
    if saved == 0:
        return False

    # Pad to k
    last_path = os.path.join(out_dir, f"frame_{saved-1:03d}.jpg")
    while saved < k:
        out_path = os.path.join(out_dir, f"frame_{saved:03d}.jpg")
        shutil.copy(last_path, out_path)
        saved += 1

    return True


def extract_audio_ffmpeg(video_path, out_wav_path, sr=16000, channels=1):
    """
    Uses ffmpeg ONLY for audio extraction.
    """
    os.makedirs(os.path.dirname(out_wav_path), exist_ok=True)
    cmd = [
        "ffmpeg", "-loglevel", "error", "-y", "-i", video_path,
        "-vn",                 # no video
        "-ac", str(channels),
        "-ar", str(sr),
        "-f", "wav",
        out_wav_path
    ]
    # Silence ffmpeg output
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return (p.returncode == 0)


def find_videos(root, exts=(".mp4", ".avi", ".mov", ".mkv")):
    paths = []
    for r, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(exts):
                paths.append(os.path.join(r, f))
    return sorted(paths)


videos = find_videos(RAVDESS_VIDEO_ROOT)
print("Found videos:", len(videos))
print("Example:", videos[0] if videos else "None")


# 3) Build dataset index + extract frames/audio + write CSV
#
# This creates the multimodal dataset in a format your pipeline can load.

records = []

for vp in tqdm(videos):
    meta = parse_ravdess_filename(vp)
    if meta is None:
        continue

    # If you want only the 6-class intersection:
    if meta["emotion"] not in ["happy", "sad", "anger", "fear", "disgust"]:
        continue

    stem = os.path.splitext(os.path.basename(vp))[0]
    clip_id = stem  # unique

    # Output locations (relative paths stored in CSV)
    frames_dir_rel = os.path.join("frames", clip_id)
    audio_rel = os.path.join("audio", f"{clip_id}.wav")

    frames_dir_abs = os.path.join(OUT_ROOT, frames_dir_rel)
    audio_abs = os.path.join(OUT_ROOT, audio_rel)

    ok_frames = extract_uniform_frames(vp, frames_dir_abs, k=FRAMES_PER_CLIP, out_size=FRAME_SIZE)
    if not ok_frames:
        continue

    if EXTRACT_AUDIO:
        ok_audio = extract_audio_ffmpeg(vp, audio_abs, sr=AUDIO_SR, channels=AUDIO_CHANNELS)
        if not ok_audio:
            continue

    records.append({
        "clip_id": clip_id,
        "video_frames_dir": frames_dir_rel,
        "audio_path": audio_rel,
        "label": meta["emotion"],
        "actor_id": meta["actor_id"],
        "gender": meta["gender"],
        "source_video": vp
    })

df = pd.DataFrame(records)
print("Prepared clips:", len(df))



# ---------------------------
# Step 4 (Modified): Fixed actor-balanced split
# Train: 16 actors (8M, 8F)
# Val:    2 actors (1M, 1F)
# Test:   6 actors (3M, 3F)
# ---------------------------

rng = np.random.default_rng(42)

# Unique actors and their genders (gender is consistent per actor in RAVDESS)
actor_gender = (
    df[["actor_id", "gender"]]
    .drop_duplicates()
    .sort_values("actor_id")
)

male_actors = actor_gender[actor_gender["gender"] == "male"]["actor_id"].tolist()
female_actors = actor_gender[actor_gender["gender"] == "female"]["actor_id"].tolist()

assert len(male_actors) == 12 and len(female_actors) == 12, "Expected 12 male and 12 female actors."

# Shuffle actors reproducibly
rng.shuffle(male_actors)
rng.shuffle(female_actors)

# Allocate actors by counts
train_actors = set(male_actors[:8] + female_actors[:8])
val_actors   = set(male_actors[8:9] + female_actors[8:9])      # 1 male, 1 female
test_actors  = set(male_actors[9:12] + female_actors[9:12])    # 3 male, 3 female

# Sanity: ensure partitions are disjoint and complete
assert len(train_actors) == 16 and len(val_actors) == 2 and len(test_actors) == 6
assert len(train_actors & val_actors) == 0
assert len(train_actors & test_actors) == 0
assert len(val_actors & test_actors) == 0
assert len(train_actors | val_actors | test_actors) == 24

# Apply split to dataframe
df["split"] = "train"
df.loc[df["actor_id"].isin(val_actors), "split"] = "val"
df.loc[df["actor_id"].isin(test_actors), "split"] = "test"

# Save updated CSV (same as before)
csv_out = os.path.join(OUT_ROOT, "mm_labels.csv")
df.to_csv(csv_out, index=False)
print("Saved with split:", csv_out)

# Quick checks
print(df["split"].value_counts())
print("\nActors per split:")
print("Train actors:", sorted(list(train_actors)))
print("Val actors:", sorted(list(val_actors)))
print("Test actors:", sorted(list(test_actors)))

print("\nGender distribution by split:")
print(df.groupby(["split", "gender"]).size().unstack(fill_value=0))

print("\nClass distribution by split:")
print(df.groupby(["split", "label"]).size().unstack(fill_value=0))

'''