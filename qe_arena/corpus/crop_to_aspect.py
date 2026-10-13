"""Center-crop an image to a target aspect ratio and report which dimension was cut. Original."""

META = {
    "entry": "crop_to_aspect",
    "origin": "Original.",
    "tags": ["aspect-ratio", "image", "crop"],
    "domain": "media-content-production",
}


def crop_to_aspect(src_w, src_h, aspect_w, aspect_h):
    """Center-crop a src_w x src_h image to the aspect_w:aspect_h shape; return [x, y, w, h, cut].

    Let wide = src_w * aspect_h and tall = aspect_w * src_h; divisions round down. If
    wide > tall the width is cut: w = src_h * aspect_w // aspect_h, h = src_h, cut "width".
    If wide < tall the height is cut: w = src_w, h = src_w * aspect_h // aspect_w, cut
    "height". Otherwise the whole image is kept, cut "none". The box is centered:
    x = (src_w - w) // 2, y = (src_h - h) // 2. Raises ValueError if any argument is below 1.
    """
    if min(src_w, src_h, aspect_w, aspect_h) < 1:
        raise ValueError("sizes must be >= 1")
    wide, tall = src_w * aspect_h, aspect_w * src_h
    if wide > tall:
        w, h, cut = src_h * aspect_w // aspect_h, src_h, "width"
    elif wide < tall:
        w, h, cut = src_w, src_w * aspect_h // aspect_w, "height"
    else:
        w, h, cut = src_w, src_h, "none"
    return [(src_w - w) // 2, (src_h - h) // 2, w, h, cut]


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    cases = [
        [1920, 1080, 16, 9],
        [1920, 1080, 4, 3],
        [1920, 1080, 9, 16],
        [4000, 3000, 16, 9],
        [1080, 1920, 1, 1],
        [1080, 1080, 1, 1],
        [2560, 1080, 64, 27],
        [1, 1, 1, 1],
        [1, 3, 2, 1],
        [3, 1, 1, 2],
        [1001, 600, 5, 3],
        [1000, 601, 5, 3],
        [0, 1080, 16, 9],
        [1920, 1080, 16, 0],
    ]
    aspects = [[16, 9], [4, 3], [1, 1], [9, 16], [21, 9], [2, 1], [3, 2], [4, 5]]
    sizes = [1, 2, 3, 5, 7, 16, 100, 333, 720, 1080, 1280, 1920, 2160, 3840]
    while len(cases) < n:
        dims = [rng.choice(sizes) if rng.random() < 0.3 else rng.randint(1, 4000) for _ in range(2)]
        aspect = rng.choice(aspects) if rng.random() < 0.7 else [rng.randint(1, 30), rng.randint(1, 30)]
        cases.append(dims + aspect)
    return cases[:n]
