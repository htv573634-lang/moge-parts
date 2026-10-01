import os
import sys
import json
import glob
import time
import numpy as np
import torch
import cv2
import trimesh

sys.path.insert(0, "MoGe")

# ── CONFIG ──
INPUT_DIR = "inputs"
OUTPUT_DIR = "outputs"
MODEL_DIR = "MoGe/checkpoints/moge-2-vitl-normal"
RESOLUTION_LEVEL = 9
IMAGE_EXTS = ("jpg", "jpeg", "jpge", "jpe", "jfif",
              "png", "bmp", "webp", "tif", "tiff", "gif", "ppm")

# Quality thresholds
MIN_SIZE = 128
BLUR_THRESHOLD = 5.0


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def find_images():
    files = []
    for ext in IMAGE_EXTS:
        for pat in (f"*.{ext}", f"*.{ext.upper()}", f"*.{ext.capitalize()}"):
            files.extend(glob.glob(os.path.join(INPUT_DIR, pat)))
    files = sorted(set(f for f in files if not os.path.basename(f).startswith(".")))
    return files


def find_checkpoint(d):
    if os.path.isfile(d):
        return d
    for c in ("model.pt", "model.pth", "moge.pt", "checkpoint.pt"):
        p = os.path.join(d, c)
        if os.path.isfile(p):
            return p
    if os.path.isdir(d):
        for f in os.listdir(d):
            if f.endswith((".pt", ".pth", ".safetensors")):
                return os.path.join(d, f)
    return None


def safe_normalize(a):
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo) if hi - lo > 1e-6 else np.zeros_like(a)


def check_image(img_rgb, name):
    """Returns (ok, reason)."""
    h, w = img_rgb.shape[:2]
    if h < MIN_SIZE or w < MIN_SIZE:
        return False, f"too small ({w}x{h}, min {MIN_SIZE})"

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur < BLUR_THRESHOLD:
        return True, f"blurry (laplacian={blur:.1f})"

    return True, f"ok (blur={blur:.1f})"


def triangulate(points, mask):
    """Build mesh from MoGe's pixel grid."""
    H, W = mask.shape
    idx = -np.ones((H, W), dtype=np.int32)
    v = mask > 0.5
    verts = points[v]
    idx[v] = np.arange(len(verts))
    faces = []
    for i in range(H - 1):
        for j in range(W - 1):
            a = idx[i, j]
            b = idx[i + 1, j]
            c = idx[i, j + 1]
            d = idx[i + 1, j + 1]
            if a >= 0 and b >= 0 and c >= 0:
                faces.append([a, b, c])
            if b >= 0 and d >= 0 and c >= 0:
                faces.append([b, d, c])
    return verts, np.array(faces, dtype=np.int32)


def process_image(model, device, img_path):
    name = os.path.splitext(os.path.basename(img_path))[0]
    out_dir = os.path.join(OUTPUT_DIR, name)
    os.makedirs(out_dir, exist_ok=True)

    # Load
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return {"name": name, "status": "error", "reason": "cannot read image"}
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H_img, W_img = img_rgb.shape[:2]

    # Quality check
    ok, reason = check_image(img_rgb, name)
    if not ok:
        return {"name": name, "status": "skipped", "reason": reason}

    log(f"  {name}: {W_img}x{H_img}  {reason}")

    # Run MoGe
    t = torch.tensor(img_rgb / 255.0, dtype=torch.float32,
                     device=device).permute(2, 0, 1)
    with torch.no_grad():
        out = model.infer(t, use_fp16=False, resolution_level=RESOLUTION_LEVEL)

    points = out["points"].cpu().numpy()
    mask = out["mask"].cpu().numpy()
    depth = out["depth"].cpu().numpy()
    normal = out["normal"].cpu().numpy() if "normal" in out else None

    H_m, W_m = mask.shape
    valid_n = int((mask > 0.5).sum())

    # Save arrays
    np.save(os.path.join(out_dir, "points.npy"), points.astype(np.float32))
    np.save(os.path.join(out_dir, "mask.npy"), mask.astype(np.uint8))
    np.save(os.path.join(out_dir, "depth.npy"), depth.astype(np.float32))
    if normal is not None:
        np.save(os.path.join(out_dir, "normals.npy"), normal.astype(np.float32))

    # Save visualizations
    mask_png = (mask * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(out_dir, "mask.png"), mask_png)

    depth_png = (safe_normalize(depth) * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(out_dir, "depth.png"), depth_png)

    if normal is not None:
        n = np.nan_to_num(normal, nan=0.0, posinf=1.0, neginf=-1.0)
        n_png = ((n + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, "normals.png"), n_png)

    # Save point cloud
    pts_flat = points.reshape(-1, 3)
    m_flat = mask.reshape(-1).astype(bool)
    valid_pts = np.nan_to_num(pts_flat[m_flat], nan=0, posinf=0, neginf=0)
    if len(valid_pts) > 0:
        preview_pts = valid_pts
        if len(preview_pts) > 1_000_000:
            idx = np.random.choice(len(preview_pts), 1_000_000, replace=False)
            preview_pts = preview_pts[idx]
        trimesh.PointCloud(preview_pts).export(os.path.join(out_dir, "points.ply"))

    # Save preview mesh
    verts, faces = triangulate(points, mask)
    if len(verts) > 0:
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        mesh.export(os.path.join(out_dir, "preview.glb"))

    # Metadata
    depth_valid = depth[mask > 0.5]
    meta = {
        "name": name,
        "source_image": os.path.basename(img_path),
        "source_size": [int(W_img), int(H_img)],
        "grid": [int(W_m), int(H_m)],
        "resolution_level": RESOLUTION_LEVEL,
        "valid_points": valid_n,
        "depth_range_m": [float(depth_valid.min()), float(depth_valid.max())],
        "bbox_world_min": [float(x) for x in points[mask > 0.5].min(0)],
        "bbox_world_max": [float(x) for x in points[mask > 0.5].max(0)],
        "quality_note": reason,
        "processed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    log(f"    -> {valid_n} valid points, grid {W_m}x{H_m}")
    return {"name": name, "status": "ok", "meta": meta}


def main():
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    images = find_images()
    if not images:
        log("No images in inputs/. Exiting.")
        return

    log(f"Found {len(images)} image(s)")

    # Load model
    ckpt = find_checkpoint(MODEL_DIR)
    if ckpt is None:
        log(f"ERROR: MoGe checkpoint not found in {MODEL_DIR}")
        sys.exit(1)
    log(f"Checkpoint: {ckpt}")

    from moge.model.v2 import MoGeModel
    device = torch.device("cpu")
    model = MoGeModel.from_pretrained(ckpt).to(device).eval().float()
    log("MoGe-2 loaded on CPU")

    # Process
    results = []
    for i, img_path in enumerate(images, 1):
        log(f"[{i}/{len(images)}] {os.path.basename(img_path)}")
        try:
            r = process_image(model, device, img_path)
        except Exception as e:
            import traceback
            traceback.print_exc()
            r = {"name": os.path.splitext(os.path.basename(img_path))[0],
                 "status": "error", "reason": str(e)}
        results.append(r)

    # Build library index
    index = {}
    for r in results:
        if r["status"] == "ok":
            index[r["name"]] = r["meta"]
    with open(os.path.join(OUTPUT_DIR, "library_index.json"), "w") as f:
        json.dump(index, f, indent=2)

    # Summary
    ok = sum(1 for r in results if r["status"] == "ok")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    failed = sum(1 for r in results if r["status"] == "error")
    log(f"\nSummary: {ok} ok, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
