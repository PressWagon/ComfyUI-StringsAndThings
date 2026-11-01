import json

import numpy as np
import torch

import folder_paths

# FIX: scipy, torchvision, Pillow, scikit-image and matplotlib are no longer
# imported at module level. The numpy/torch paths below replace scipy,
# torchvision and Pillow entirely; scikit-image and matplotlib are imported
# lazily inside FourierAnalysisNode, so a missing optional dependency can no
# longer stop the string nodes from loading.

YELLOW = "\033[1;33m"
CYAN = "\033[1;36m"
RESET = "\033[0m"


class AnyType(str):
    """A special type that can be connected to any other types. Credit to pythongosssss"""

    def __ne__(self, __value: object) -> bool:
        return False


any_type = AnyType("*")


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------

def _s(value):
    """Coerce None to "" so string concatenation can never raise TypeError."""
    return "" if value is None else str(value)


def _normalize_to_255(arr):
    """Scale an array to 0-255. A flat array returns zeros instead of NaNs."""
    # FIX: the original lambda divided by (max - min), giving NaN on flat images.
    arr = np.asarray(arr, dtype=np.float64)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo) * 255.0


def _as_rgb(img_np):
    """HWC float array -> HWC array with exactly three channels."""
    channels = img_np.shape[-1]
    if channels == 1:
        return np.repeat(img_np, 3, axis=-1)
    if channels >= 3:
        return img_np[..., :3]
    raise ValueError(f"Expected an image with 1, 3 or 4 channels, got {channels}.")


# ---------------------------------------------------------------------------
# Part 1 - String formatting
# ---------------------------------------------------------------------------

# Picks a Lora as an output to both send to a lora loader and the metadata as a string
class PWLoraSelector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"),),  # List of LoRA filenames
            }
        }

    CATEGORY = 'Strings&Things'
    # NOTE: RETURN_TYPES is evaluated once at import, so LoRAs added later only
    # appear in this output's type after a ComfyUI restart. It is kept as a
    # combo list so the output can still connect to a LoRA loader's combo input.
    RETURN_TYPES = (folder_paths.get_filename_list("loras"), "STRING",)
    RETURN_NAMES = ("LORA_NAME", "metadata",)
    FUNCTION = "get_lora_name"

    def get_lora_name(self, lora_name):
        output = str(lora_name)
        return (output, output)


# Takes up to 5 Lora names. Chaining works because list inputs are flattened.
# FIX: removed the stale "TO DO: fix chaining" comment; the flatten below is the fix.
class PWLoraNameCollector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "Lora_A": ("STRING",),
                "Lora_B": ("STRING",),
                "Lora_C": ("STRING",),
                "Lora_D": ("STRING",),
                "Lora_E": ("STRING",),
            }
        }

    CATEGORY = 'Strings&Things'
    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("LORA_NAMES",)
    FUNCTION = "LoraNameConcat"

    def LoraNameConcat(self, Lora_A=None, Lora_B=None, Lora_C=None, Lora_D=None, Lora_E=None):
        lora_names = []
        for item in (Lora_A, Lora_B, Lora_C, Lora_D, Lora_E):
            if not item:  # skip None and empty strings / lists
                continue
            if isinstance(item, (list, tuple)):
                lora_names.extend(name for name in item if name)  # flatten chained collectors
            else:
                lora_names.append(item)
        return (lora_names,)


# A node for printing string data to the console
class DebugString:
    CATEGORY = "Strings&Things"
    FUNCTION = "PrintString"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # NOTE: the input name (and so the method parameter below) must
                # stay "DebugString" to keep existing workflows working.
                "DebugString": (any_type,),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("DebugOutput",)  # Optional output, maybe it'll come in handy some day.
    OUTPUT_NODE = True  # Kept: this node exists for its console side effect.

    def PrintString(self, DebugString):
        text = str(DebugString)
        print(f"{YELLOW}DebugString: {text}{RESET}")
        return (text,)  # Return the input string for potential chaining


class FormatConcatStrings:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "separator": ("STRING", {"multiline": False, "default": " "}),
            },
            "optional": {
                "A_Pre_string": ("STRING", {"multiline": False}),
                "A_Body_string": ("STRING", {"multiline": False}),
                "A_Post_string": ("STRING", {"multiline": False}),
                "B_Pre_string": ("STRING", {"multiline": False}),
                "B_Body_string": ("STRING", {"multiline": False}),
                "B_Post_string": ("STRING", {"multiline": False}),
                "C_Pre_string": ("STRING", {"multiline": False}),
                "C_Body_string": ("STRING", {"multiline": False}),
                "C_Post_string": ("STRING", {"multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("FormattedString",)
    FUNCTION = "triple_format_concat_strings"
    # FIX: OUTPUT_NODE removed. It forced this node to run even when nothing
    # consumed its output. Re-add it if you rely on it as a terminal node.
    CATEGORY = "Strings&Things"

    def triple_format_concat_strings(self, separator=" ",
                                     A_Pre_string="", A_Body_string="", A_Post_string="",
                                     B_Pre_string="", B_Body_string="", B_Post_string="",
                                     C_Pre_string="", C_Body_string="", C_Post_string=""):
        groups = (
            (A_Pre_string, A_Body_string, A_Post_string),
            (B_Pre_string, B_Body_string, B_Post_string),
            (C_Pre_string, C_Body_string, C_Post_string),
        )
        parts = []
        for pre, body, post in groups:
            pre, body, post = _s(pre), _s(body), _s(post)
            if body:  # empty bodies drop their pre/post text too
                parts.append(pre + body + post)
        return (_s(separator).join(parts),)


# A node for adding pre and post text. Will only output if Body is not empty.
class FormattingSingle:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "PRE_string": ("STRING", {"multiline": False}),
                "BODY_string": ("STRING", {"multiline": False}),
                "POST_string": ("STRING", {"multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("FormattedString",)
    FUNCTION = "format_single"
    # FIX: OUTPUT_NODE removed (see FormatConcatStrings).
    CATEGORY = "Strings&Things"

    def format_single(self, PRE_string="", BODY_string="", POST_string=""):
        body = _s(BODY_string)
        if not body:
            return ("",)
        return (_s(PRE_string) + body + _s(POST_string),)


"""
Part 2
Weird nodes that may or may not be useful

-Image mosaic censor node
-Fourier analysis node
-Cosine Similarity & Euclidean Distance between text embeddings
-Image difference comparison tool
"""


def mosaic_numpy(arr, tile_size):
    """
    Median-colour mosaic of one HxWxC float array.

    FIX: vectorised (the original called putpixel once per pixel in Python) and
    edge tiles are padded with NaN and ignored by nanmedian. The original used
    PIL's crop() past the image border, which pads with black and skewed the
    median of partial edge tiles.
    """
    tile_size = max(1, int(tile_size))
    h, w, c = arr.shape
    pad_h = (-h) % tile_size
    pad_w = (-w) % tile_size
    padded = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)),
                    mode="constant", constant_values=np.nan)
    tiles = padded.reshape((h + pad_h) // tile_size, tile_size,
                           (w + pad_w) // tile_size, tile_size, c)
    medians = np.nanmedian(tiles, axis=(1, 3))
    mosaic = np.repeat(np.repeat(medians, tile_size, axis=0), tile_size, axis=1)
    return np.ascontiguousarray(mosaic[:h, :w], dtype=np.float32)


# Applies a mosaic effect to the entire image. Can be composited back onto the original
class MosaicEffectNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "tile_size": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("MosaicImage",)
    FUNCTION = "apply_mosaic"
    CATEGORY = "Strings&Things/Extras"
    # FIX: OUTPUT_NODE removed; it forced the node to run with nothing connected.

    def apply_mosaic(self, image, tile_size=10):
        # FIX: processes every image in the batch (the original silently kept only the first)
        # and stays in float instead of round-tripping through 8-bit PIL.
        results = [
            torch.from_numpy(mosaic_numpy(img.detach().cpu().numpy().astype(np.float32), tile_size))
            for img in image
        ]
        return (torch.stack(results),)


def radial_entropy(magnitude):
    """
    Shannon entropy (bits) of the value histogram at each integer radius from
    the centre of a 2D array holding values in 0-255.

    FIX: vectorised with one bincount. The original looped over every radius and
    built a full-image mask each time (roughly O(radii x pixels)). Binning
    matches np.histogram(bins=256, range=(0, 255)).
    """
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.int64)
    max_r = int(r.max())
    bins = np.clip((magnitude / 255.0 * 256).astype(np.int64), 0, 255)
    counts = np.bincount((r * 256 + bins).ravel(),
                         minlength=(max_r + 1) * 256).reshape(max_r + 1, 256).astype(np.float64)
    totals = counts.sum(axis=1, keepdims=True)
    p = counts / np.maximum(totals, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(p > 0, np.log2(p), 0.0)
    return -(p * logp).sum(axis=1) + 0.0  # + 0.0 turns -0.0 into 0.0


def fourier_spectrum(rgb):
    """HxWx3 float array -> list of three 0-255 log-magnitude spectra."""
    spectra = []
    for c in range(3):
        shifted = np.fft.fftshift(np.fft.fft2(rgb[..., c]))
        spectra.append(_normalize_to_255(np.log1p(np.abs(shifted))))
    return spectra


class FourierAnalysisNode:
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("Fourier Image", "Entropy Heatmap")
    FUNCTION = "FAnalysis"
    OUTPUT_NODE = True
    CATEGORY = "Strings&Things/Extras"

    def FAnalysis(self, image):
        # FIX: lazy imports so the rest of the pack loads without these installed.
        try:
            import matplotlib
            from skimage.filters.rank import entropy as local_entropy
            from skimage.morphology import disk
        except ImportError as exc:
            raise RuntimeError(
                "Fourier Analysis needs scikit-image and matplotlib. "
                "Run: pip install scikit-image matplotlib"
            ) from exc

        colormap = matplotlib.colormaps["viridis"]
        fourier_out, heatmap_out = [], []

        # FIX: loops over the batch (the original used squeeze(0), which breaks for batch > 1).
        for img in image:
            rgb = _as_rgb(img.detach().cpu().numpy().astype(np.float32))

            # --- Fourier magnitude spectrum ---
            spectra = fourier_spectrum(rgb)
            radial_avg = sum(radial_entropy(m) for m in spectra) / 3.0
            print(f"{CYAN}Radial Entropy Avg:{RESET} {radial_avg[:20]}...")
            fourier_u8 = np.stack(spectra, axis=-1).astype(np.uint8)

            # --- Local entropy heatmap ---
            # FIX: removed the unused global entropy / entropy_avg calculation.
            rgb_u8 = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
            local = [
                _normalize_to_255(local_entropy(rgb_u8[..., c], disk(5)))
                for c in range(3)
            ]
            avg_map = (sum(local) / 3.0).astype(np.uint8)
            # Calling a colormap with ints indexes its 256-entry lookup table directly.
            heat_u8 = (colormap(avg_map)[..., :3] * 255).astype(np.uint8)

            fourier_out.append(torch.from_numpy(fourier_u8.astype(np.float32) / 255.0))
            heatmap_out.append(torch.from_numpy(heat_u8.astype(np.float32) / 255.0))

        return (torch.stack(fourier_out), torch.stack(heatmap_out))


# Takes two text inputs and two CLIP inputs and calculates the Cosine Similarity and Euclidean distance between them
class TextEmbeddingsInterrogator:
    FUNCTION = "interrogation"
    CATEGORY = "Strings&Things/Extras"
    DESCRIPTION = "Calculates the Cosine Similarity and Euclidean Distance between two text embeddings"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Output string",)

    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "CLIP_1": ("CLIP",),
                "Text_1": ("STRING", {"default": ""}),
                "CLIP_2": ("CLIP",),
                "Text_2": ("STRING", {"default": ""}),
            }
        }

    def get_text_embeddings(self, clip, text):
        tokens = clip.tokenize(text)
        # FIX: use the public pooled-output API instead of reaching into
        # encode_from_tokens_scheduled()[0][1]["pooled_output"].
        _, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        if pooled is None:
            raise ValueError(
                "This CLIP model did not return a pooled output, so its embedding cannot be compared."
            )
        return pooled[:1].float().flatten()  # first prompt only, 1-D vector

    def interrogation(self, Text_1, Text_2, CLIP_1, CLIP_2):
        with torch.no_grad():
            embedding1 = self.get_text_embeddings(CLIP_1, Text_1)
            embedding2 = self.get_text_embeddings(CLIP_2, Text_2)

            # FIX: readable error instead of an opaque torch.mm shape failure.
            if embedding1.shape != embedding2.shape:
                raise ValueError(
                    f"Embedding sizes differ ({embedding1.numel()} vs {embedding2.numel()}). "
                    "Both CLIP models must produce pooled embeddings of the same dimension."
                )

            cos_sim_value = torch.nn.functional.cosine_similarity(embedding1, embedding2, dim=0).item()
            # Euclidean distance is computed on the raw (un-normalised) embeddings.
            distance_value = torch.norm(embedding1 - embedding2, p=2).item()

        print(f'{YELLOW}Text: "{Text_1}", "{Text_2}"{RESET}')
        print(f"{YELLOW}CosSim: {cos_sim_value}{RESET}")
        print(f"{YELLOW}Euclidean Distance: {distance_value}{RESET}")

        output_string = f"Cosine Similarity: {cos_sim_value}, Euclidean Distance: {distance_value}"
        # FIX: must return a tuple. Returning a bare string made ComfyUI unpack it
        # per character, so the connected output would have been just "C".
        return (output_string,)


# Takes two images and outputs an image of the pixel-wise difference
class ImageDifference:
    FUNCTION = "ImageDiff"
    CATEGORY = "Strings&Things/Extras"
    DESCRIPTION = "Pairwise image comparison"  # FIX: typo
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "contrast_boost": ("BOOLEAN", {"default": False}),
                "console_mse": ("BOOLEAN", {"default": True}),
            }
        }

    def ImageDiff(self, image1, image2, contrast_boost, console_mse):
        # FIX: explicit, readable validation instead of an obscure broadcast error.
        if image1.shape[1:3] != image2.shape[1:3]:
            raise ValueError(
                f"Images must have the same height and width, got "
                f"{tuple(image1.shape[2:0:-1])} and {tuple(image2.shape[2:0:-1])} (width x height)."
            )
        b1, b2 = image1.shape[0], image2.shape[0]
        if b1 != b2 and 1 not in (b1, b2):
            raise ValueError(f"Batch sizes are incompatible: {b1} vs {b2}.")

        # Compare only the channels both images share (e.g. RGB vs RGBA).
        channels = min(image1.shape[-1], image2.shape[-1])
        a, b = image1[..., :channels], image2[..., :channels]

        # FIX: no squeeze/unsqueeze. Subtraction already broadcasts over [B, H, W, C],
        # so batches work and the output stays 4-D.
        diff_image = torch.abs(a - b)  # magnitude matters more than direction here

        if contrast_boost:
            boost_factor = 5.0  # can make this a user input later
            diff_image = (diff_image * boost_factor).clamp(0, 1)

        if console_mse:
            mse_percent = torch.mean((a - b) ** 2).item() * 100
            print(f"{YELLOW}Mean Square Error: {mse_percent:.2f}%{RESET}")

        return (diff_image,)


'''
Part 3 MOSS-Image

-MOSS to ComfyUI BBoxes
'''


# Converts the Moss-Image Layout2Image json format into ComfyUI's Create Bounding Boxes format
class MOSSToComfyUIBBoxes:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "moss_json": ("STRING", {"multiline": True, "default": ""}),
                "source_width": ("INT", {"default": 1024, "min": 1, "max": 16384}),
                "source_height": ("INT", {"default": 1024, "min": 1, "max": 16384}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("bboxes_json", "summary")
    OUTPUT_TOOLTIPS = (
        "Converted bounding-box elements in ComfyUI's 0-1000 coordinate format. Connect to the Create Bounding Boxes node's bboxes input.",
        "The summary section extracted from the prompt, can be connected to the MOSS-Image Layout prompt input to reproduce original image.",
    )
    FUNCTION = "convert"
    CATEGORY = "MOSS/Utilities"

    def convert(self, moss_json, source_width, source_height):
        if not moss_json or not moss_json.strip():
            # FIX: "" (was "{}"), consistent with the plain-text summary returned below.
            return ("", "")

        try:
            data = json.loads(moss_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid MOSS JSON: {exc}") from exc

        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object containing an items array.")

        coord_space = data.get("bbox_coord_space")
        if coord_space != "abs_centralized":
            # FIX: include the value actually received.
            raise ValueError(f"Expected bbox_coord_space='abs_centralized', got {coord_space!r}.")

        bbox_format = data.get("bbox_format", "yxyx")
        if bbox_format != "yxyx":
            raise ValueError(f"Only bbox_format='yxyx' is currently supported, got {bbox_format!r}.")

        items = data.get("items")
        if not isinstance(items, list):
            raise ValueError("The MOSS JSON does not contain a valid items array.")

        converted = []
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                raise ValueError(f"Item {index} is not a JSON object.")

            bbox = item.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                raise ValueError(
                    f"Item {index} has an invalid bbox. Expected [ymin, xmin, ymax, xmax]."
                )

            try:
                ymin, xmin, ymax, xmax = map(float, bbox)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Item {index} contains non-numeric bbox coordinates.") from exc

            converted_bbox = [
                round((ymin + source_height / 2) / source_height * 1000),
                round((xmin + source_width / 2) / source_width * 1000),
                round((ymax + source_height / 2) / source_height * 1000),
                round((xmax + source_width / 2) / source_width * 1000),
            ]
            converted_bbox = [max(0, min(1000, value)) for value in converted_bbox]

            label = str(item.get("label", "object")).strip().lower()  # FIX: case-insensitive
            element = {
                "type": "text" if label == "text" else "obj",
                "bbox": converted_bbox,
                "desc": item.get("caption", ""),
                "color_palette": item.get("palette", []),
            }
            if element["type"] == "text":
                element["text"] = item.get("text", "")

            converted.append(element)

        bboxes_json = json.dumps(converted, ensure_ascii=False, indent=4)

        # Extract the summary caption as plain text.
        # FIX: no crash when "summary" is missing, null or not an object.
        summary_block = data.get("summary")
        summary = summary_block.get("caption", "") if isinstance(summary_block, dict) else ""

        return (bboxes_json, summary)


NODE_CLASS_MAPPINGS = {
    "PWLoraSelector": PWLoraSelector,
    "PWLoraNameCollector": PWLoraNameCollector,
    "DebugString": DebugString,
    "FormatConcatStrings": FormatConcatStrings,
    "FormattingSingle": FormattingSingle,
    "MosaicEffectNode": MosaicEffectNode,
    "FourierAnalysisNode": FourierAnalysisNode,
    "TextEmbeddingsInterrogator": TextEmbeddingsInterrogator,
    "ImageDifference": ImageDifference,
    "MOSSToComfyUIBBoxes": MOSSToComfyUIBBoxes,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PWLoraSelector": "Lora Selector",
    "PWLoraNameCollector": "Lora Name Collector",
    "DebugString": "🔧 Debug String",
    "FormatConcatStrings": "Formatting and Concatenating Strings",
    "FormattingSingle": "Formatting Single String",
    "MosaicEffectNode": "Apply Mosaic Effect",
    "FourierAnalysisNode": "Fourier Analysis",
    "TextEmbeddingsInterrogator": "Text Embeddings Interrogator",
    "ImageDifference": "Image Difference",
    "MOSSToComfyUIBBoxes": "MOSS to ComfyUI BBoxes",
}
