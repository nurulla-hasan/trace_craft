"""Experiment: does local PCA contrast stretch expose whale baleen lines inside the belly?"""
import cv2
import numpy as np
from engine.preprocess import load_image, extract_background_mask, split_foreground_regions

SRC = "C:/Users/golap/.gemini/antigravity/brain/1f9101d7-2561-446b-9ca5-618bee8fc2f7/.user_uploaded/media_1795061237693_b887cd3b.jpg"
OUT = "engine/test_outputs/belly_contrast_experiment.png"

img = load_image(SRC)
bg_mask, bg_color = extract_background_mask(img, bg_tol=14)
regs = split_foreground_regions(img, bg_mask, bg_color)

rows = []
for wi in range(2):
    r = regs[wi]
    crop = r["crop_bgr"]
    fg = ~r["crop_bg_mask"]
    h, w = crop.shape[:2]
    f = crop.astype(np.float32)

    # Belly = light-blue pixels (brightness above 150 mean) deep inside the silhouette
    inner = cv2.erode(fg.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    lum = f.mean(axis=2)
    belly = inner & (lum > 125)
    n_belly = int(belly.sum())
    if n_belly < 30:
        print(f"whale {wi+1}: belly too small ({n_belly})")
        continue

    # PCA on belly pixel colors -> 1D axis of max variation (where the lines live)
    px = f[belly]
    mean = px.mean(axis=0)
    cov = np.cov((px - mean).T)
    evals, evecs = np.linalg.eigh(cov)
    axis = evecs[:, -1]
    proj = (f - mean) @ axis
    p_lo, p_hi = np.percentile(proj[belly], [1, 99])
    stretched = np.clip((proj - p_lo) / max(p_hi - p_lo, 1e-6), 0, 1)
    stretched[~belly] = 0.5

    # 4x upscale, then DoG ridge detection on the stretched channel
    s4 = cv2.resize(stretched, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)
    fine = cv2.GaussianBlur(s4, (0, 0), 1.4)
    coarse = cv2.GaussianBlur(s4, (0, 0), 8.0)
    dog = fine - coarse
    dark_lines = (dog < -0.06)  # lines darker than their surroundings
    mask4 = cv2.resize(belly.astype(np.uint8), (w * 4, h * 4), interpolation=cv2.INTER_NEAREST) > 0
    mask4 = cv2.erode(mask4.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    det = dark_lines & mask4

    orig4 = cv2.resize(crop, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)
    str4 = cv2.cvtColor((s4 * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    det4 = cv2.cvtColor((det * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    row = np.hstack([orig4, str4, det4])
    cv2.putText(row, f"whale {wi+1}: original | PCA-stretched belly | detected lines ({int(det.sum())} px)",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    rows.append(row)
    print(f"whale {wi+1}: belly px={n_belly}, pca var ratio={evals[-1]/evals.sum():.2f}, detected={int(det.sum())}")

width = max(r.shape[1] for r in rows)
rows = [cv2.copyMakeBorder(r, 0, 6, 0, width - r.shape[1], cv2.BORDER_CONSTANT, value=(60, 60, 60)) for r in rows]
cv2.imwrite(OUT, np.vstack(rows))
print("saved", OUT)
