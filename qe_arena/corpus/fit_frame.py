"""Letterbox or center-crop an image into a frame of another aspect ratio. Original."""

META = {
    "entry": "fit_frame",
    "origin": "Original.",
    "tags": ["aspect-ratio", "image", "layout"],
    "domain": "media-content-production",
}


def fit_frame(src_w, src_h, dst_w, dst_h, mode):
    """Place a src_w x src_h image in a dst_w x dst_h frame; return {"box": [x, y, w, h], "adjust": str}.

    The image is wider than the frame when src_w * dst_h > dst_w * src_h, narrower when it
    is less, and the same shape when equal. All divisions round down.
    mode "fit" scales the whole image to fit inside the frame; the box is in frame
    coordinates: a wider image gets w = dst_w, h = src_h * dst_w // src_w and adjust
    "letterbox"; a narrower one gets h = dst_h, w = src_w * dst_h // src_h and adjust
    "pillarbox"; the same shape fills the frame with adjust "none". Centered:
    x = (dst_w - w) // 2 and y = (dst_h - h) // 2.
    mode "fill" crops the source to the frame's shape; the box is in source coordinates: a
    wider image keeps h = src_h with w = src_h * dst_w // dst_h and adjust "crop-sides"; a
    narrower one keeps w = src_w with h = src_w * dst_h // dst_w and adjust "crop-top-bottom";
    the same shape keeps the whole source with adjust "none". Centered:
    x = (src_w - w) // 2 and y = (src_h - h) // 2.
    Raises ValueError if any size is below 1 or mode is neither "fit" nor "fill".
    """
    if min(src_w, src_h, dst_w, dst_h) < 1:
        raise ValueError("sizes must be at least 1")
    if mode != "fit" and mode != "fill":
        raise ValueError("mode must be fit or fill")
    wide, tall = src_w * dst_h, dst_w * src_h
    if mode == "fit":
        if wide > tall:
            w, h, adjust = dst_w, src_h * dst_w // src_w, "letterbox"
        elif wide < tall:
            w, h, adjust = src_w * dst_h // src_h, dst_h, "pillarbox"
        else:
            w, h, adjust = dst_w, dst_h, "none"
        return {"box": [(dst_w - w) // 2, (dst_h - h) // 2, w, h], "adjust": adjust}
    if wide > tall:
        w, h, adjust = src_h * dst_w // dst_h, src_h, "crop-sides"
    elif wide < tall:
        w, h, adjust = src_w, src_w * dst_h // dst_w, "crop-top-bottom"
    else:
        w, h, adjust = src_w, src_h, "none"
    return {"box": [(src_w - w) // 2, (src_h - h) // 2, w, h], "adjust": adjust}


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    cases = [
        [1920, 1080, 1280, 720, "fit"],
        [1920, 1080, 1280, 720, "fill"],
        [4000, 3000, 1920, 1080, "fit"],
        [4000, 3000, 1920, 1080, "fill"],
        [1080, 1920, 1920, 1080, "fit"],
        [1080, 1920, 1920, 1080, "fill"],
        [2560, 1080, 1280, 720, "fit"],
        [2560, 1080, 1280, 720, "fill"],
        [1, 1, 1, 1, "fit"],
        [1, 3, 2, 1, "fill"],
        [3, 1, 1, 2, "fit"],
        [0, 1080, 1280, 720, "fit"],
        [1920, 1080, 1280, 0, "fill"],
        [1920, 1080, 1280, 720, "stretch"],
    ]
    sizes = [1, 2, 3, 5, 7, 16, 100, 333, 720, 1080, 1280, 1920, 2160, 3840]
    while len(cases) < n:
        dims = [rng.choice(sizes) if rng.random() < 0.3 else rng.randint(1, 4000) for _ in range(4)]
        cases.append(dims + [rng.choice(["fit", "fill"])])
    return cases[:n]
