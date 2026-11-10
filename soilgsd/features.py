"""Path B inputs: (i) embedding, (ii) texture, (iii) colour, (iv) coarse stats."""
import numpy as np
import cv2


def center_crop(img, frac):
    H, W = img.shape[:2]
    h, w = int(H * frac), int(W * frac)
    y, x = (H - h) // 2, (W - w) // 2
    return img[y:y + h, x:x + w]


def resample_to(img, rho, target):
    s = target / rho
    if abs(s - 1) < 1e-3:
        return img
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)


def texture_feats(rgb) -> np.ndarray:
    """Rubin-style autocorrelation decay + DoG band energies (wavelet stand-in)."""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    g -= g.mean()
    F = np.fft.rfft2(g)
    ac = np.fft.irfft2(F * np.conj(F), s=g.shape)
    ac /= ac[0, 0] + 1e-9
    ac_f = [ac[0, l] for l in (1, 2, 4, 8, 16)]
    en, prev = [], g
    for s in (1, 2, 4, 8, 16):
        cur = cv2.GaussianBlur(g, (0, 0), s)
        en.append(np.log1p(((prev - cur) ** 2).mean())); prev = cur
    return np.array(ac_f + en)


def colour_feats(rgb) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    return np.r_[lab.mean(0), lab.std(0)]


class SimpleEmbedder:
    """Placeholder for DINOv2: tiny colour-layout vector. Swap with Dinov2Embedder."""
    def __call__(self, rgb):
        return cv2.resize(rgb, (4, 4), interpolation=cv2.INTER_AREA).astype(np.float32).ravel() / 255.0


class Dinov2Embedder:
    def __init__(self, name="dinov2_vits14", device="cpu"):
        import torch
        self.torch = torch
        self.m = torch.hub.load("facebookresearch/dinov2", name).to(device).eval()
        self.device = device

    def __call__(self, rgb, tile=224):
        t = self.torch
        H, W = rgb.shape[:2]
        embs = []
        for y in range(0, max(H - tile, 0) + 1, tile):
            for x in range(0, max(W - tile, 0) + 1, tile):
                p = rgb[y:y + tile, x:x + tile]
                if p.shape[:2] != (tile, tile): continue
                x_ = (torch.from_numpy(p).float().permute(2, 0, 1) / 255 - torch.tensor([.485, .456, .406])[:, None, None]) \
                     / torch.tensor([.229, .224, .225])[:, None, None]
                with t.no_grad():
                    embs.append(self.m(x_[None].to(self.device)).cpu().numpy()[0])
        return np.mean(embs, 0)


def image_feats(rgb, embedder) -> np.ndarray:
    return np.r_[embedder(rgb), texture_feats(rgb), colour_feats(rgb)]
