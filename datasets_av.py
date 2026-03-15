import json, random, math
import torch
import torchaudio
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from pathlib import Path

class AVManifestDataset(Dataset):
    """
    manifest.jsonl created by preprocessing.
    Loads:
      - K frames from frames_dir
      - audio wav
    """
    def __init__(self, manifest_jsonl, split, label_map, num_frames=8, sr=16000, seconds=4.0):
        self.items = []
        with open(manifest_jsonl, "r") as f:
            for line in f:
                rec = json.loads(line)
                if rec["split"] == split:
                    self.items.append(rec)

        self.label_map = label_map
        self.num_frames = num_frames
        self.sr = sr
        self.max_len = int(sr * seconds)

        self.tf_img = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2,0.2,0.2,0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

    def _load_frames(self, frames_dir):
        frames = sorted([str(p) for p in Path(frames_dir).glob("*.jpg")])
        if len(frames) == 0:
            # fallback: a dummy tensor
            return torch.zeros(self.num_frames, 3, 224, 224)

        # sample uniformly
        idxs = torch.linspace(0, len(frames)-1, steps=self.num_frames).long().tolist()
        out = []
        for i in idxs:
            img = Image.open(frames[i]).convert("RGB")
            out.append(self.tf_img(img))
        return torch.stack(out, dim=0)  # [T,3,224,224]

    def _load_audio(self, wav_path):
        wav, sr = torchaudio.load(wav_path)  # [1, N] usually
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        wav = wav.mean(dim=0)  # [N]
        # crop/pad to max_len
        if wav.numel() >= self.max_len:
            start = random.randint(0, wav.numel() - self.max_len)
            wav = wav[start:start+self.max_len]
        else:
            pad = self.max_len - wav.numel()
            wav = torch.nn.functional.pad(wav, (0,pad))
        return wav  # [max_len]

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        rec = self.items[idx]
        frames = self._load_frames(rec["frames_dir"])
        audio = self._load_audio(rec["audio_wav"])
        y = self.label_map[rec["emotion"]]
        return {"frames": frames, "audio": audio, "label": y, "id": rec["id"]}