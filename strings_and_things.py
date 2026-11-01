import folder_paths

import torch
from torchvision.transforms import ToPILImage, ToTensor
from scipy.fftpack import fft2, fftshift
from scipy.stats import entropy
from PIL import Image
import numpy as np
from skimage.filters.rank import entropy as local_entropy
from skimage.morphology import disk
import matplotlib.cm as cm
import json


class AnyType(str):
    """A special type that can be connected to any other types. Credit to pythongosssss"""

    def __ne__(self, __value: object) -> bool:
        return False


any_type = AnyType("*")

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
    RETURN_TYPES = (folder_paths.get_filename_list("loras"), "STRING",)
    RETURN_NAMES = ("LORA_NAME", "metadata",)
    FUNCTION = "get_lora_name"

    def get_lora_name(self, lora_name):
        output = str(lora_name)
        return (output, output)

# Takes up to 5 Lora names. Is not setup for chaining multiple PWLoraNameCollector nodes together (TO DO: fix this, chaining creates multiple lists)
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
        # Collect all inputs into a list
        inputs = [Lora_A, Lora_B, Lora_C, Lora_D, Lora_E]

        # Flatten inputs to handle cases where any of them might already be a list
        lora_names = []
        for item in inputs:
            if item:  # Ignore None values
                if isinstance(item, list):
                    lora_names.extend(item)  # Unpack list items
                else:
                    lora_names.append(item)  # Append single string

        return (lora_names,)

 # A node for printing string data to the console
class DebugString:
    CATEGORY = "Strings&Things"
    FUNCTION = "PrintString"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "DebugString": (any_type,),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("DebugOutput",)  # Optional output, maybe it'll come in handy some day.
    OUTPUT_NODE = True

    def PrintString(self, DebugString):
        DebugString=str(DebugString)
        print(f"\033[1;33mDebugString: {DebugString}\033[0m")  # Output the string to the console for debugging
        return (DebugString,)  # Return the input string for potential chaining

class FormatConcatStrings:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "separator": ("STRING", {
                    "multiline": False,
                    "default": " ",
                }),
            },
            "optional": {
                "A_Pre_string": ("STRING", {
                    "multiline": False,
                }),
                "A_Body_string": ("STRING", {
                    "multiline": False,
                }),
                "A_Post_string": ("STRING", {
                    "multiline": False,
                }),
                "B_Pre_string": ("STRING", {
                    "multiline": False,
                }),
                "B_Body_string": ("STRING", {
                    "multiline": False,
                }),
                "B_Post_string": ("STRING", {
                    "multiline": False,
                }),
                "C_Pre_string": ("STRING", {
                    "multiline": False,
                }),
                "C_Body_string": ("STRING", {
                    "multiline": False,
                }),
                "C_Post_string": ("STRING", {
                    "multiline": False,
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("FormattedString",)
    FUNCTION = "triple_format_concat_strings"
    OUTPUT_NODE = True
    CATEGORY = "Strings&Things"

    def triple_format_concat_strings(self, separator=" ", A_Pre_string="", A_Body_string="", A_Post_string="", B_Pre_string="", B_Body_string="", B_Post_string="", C_Pre_string="", C_Body_string="", C_Post_string=""):
        A_part = (A_Pre_string + A_Body_string + A_Post_string) if A_Body_string else ""
        B_part = (B_Pre_string + B_Body_string + B_Post_string) if B_Body_string else ""
        C_part = (C_Pre_string + C_Body_string + C_Post_string) if C_Body_string else ""
        formatted_string = separator.join(filter(None, [A_part, B_part, C_part]))
        return (formatted_string,)

# A node for adding pre and post text. Will only output if Body is not empty.
class FormattingSingle:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "PRE_string": ("STRING", {
                    "multiline": False,
                }),
                "BODY_string": ("STRING", {
                    "multiline": False,
                }),
                "POST_string": ("STRING", {
                    "multiline": False,
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("FormattedString",)
    FUNCTION = "format_single"
    OUTPUT_NODE = True
    CATEGORY = "Strings&Things"


    def format_single (self, PRE_string="", BODY_string="", POST_string=""):
        if not BODY_string:  # BODY_string is empty or None to avoid formatting empty strings
            formatted_string = ""
        else:
            formatted_string = PRE_string + BODY_string + POST_string
        return (formatted_string,)

""" Part 2
    Weird nodes that may or may not be useful
        -Image mosaic censor node
        -Fourier analysis node
        -[SDXL & 1.5 Only for now] Cosine Similarity & Euclidean Distance
        -Image difference comparison tool
"""

# Applies a mosaic effect to the entire image. Can be composited back onto the original
class MosaicEffectNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "tile_size": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("MosaicImage",)
    FUNCTION = "apply_mosaic"
    CATEGORY = "Strings&Things/Extras"
    OUTPUT_NODE = True

    def apply_mosaic(self, image, tile_size=10):
        # Rearrange the tensor from Comfy's [batch_size, height, width, channels] format to PIL's [batch_size, channels, height, width] format
        image = image.permute(0, 3, 1, 2)

        # Convert the tensor to a PIL image (processing only the first image in the batch)
        image = ToPILImage()(image[0])

        # Get image dimensions
        width, height = image.size

        # Create a new image for the output (same size)
        mosaic_image = Image.new("RGB", (width, height))

        # Process the image in tiles
        for y in range(0, height, tile_size):
            for x in range(0, width, tile_size):
                # Define the tile box
                box = (x, y, x + tile_size, y + tile_size)

                # Crop the tile from the image
                tile = image.crop(box)

                # Get the median color of the tile
                median_color = self.get_median_color(tile)

                # Fill the tile area in the output image with the median color
                for yy in range(y, min(y + tile_size, height)):
                    for xx in range(x, min(x + tile_size, width)):
                        mosaic_image.putpixel((xx, yy), median_color)

        # Convert the PIL image back to tensor
        mosaic_image_tensor = ToTensor()(mosaic_image).unsqueeze(0)  # Add batch dimension back
        mosaic_image_tensor = mosaic_image_tensor.permute(0, 2, 3, 1)

        return (mosaic_image_tensor,)

    @staticmethod
    def get_median_color(tile):
        # Convert the tile to a numpy array
        pixels = np.array(tile)

        # Calculate the median color for each channel
        r = np.median(pixels[:, :, 0])
        g = np.median(pixels[:, :, 1])
        b = np.median(pixels[:, :, 2])

        return int(r), int(g), int(b)


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

    @staticmethod
    def calculate_entropy(image_channel):
        hist, _ = np.histogram(image_channel.flatten(), bins=256, range=(0, 255), density=True)
        hist = hist[hist > 0]
        return entropy(hist, base=2)

    @staticmethod
    def radial_entropy(magnitude):
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        y, x = np.indices((h, w))
        r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(np.int32)

        max_r = np.max(r)
        entropy_values = np.zeros(max_r + 1)

        for radius in range(max_r + 1):
            mask = (r == radius)
            if np.any(mask):
                values = magnitude[mask]
                hist, _ = np.histogram(values, bins=256, range=(0, 255), density=True)
                hist = hist[hist > 0]
                entropy_values[radius] = entropy(hist, base=2)

        return entropy_values

    def FAnalysis(self, image):
        image = image.permute(0, 3, 1, 2)  # [B, C, H, W]
        image = image.squeeze(0)          # [C, H, W]

        # Fourier Transform
        fft_r = fft2(image[0].numpy())
        fft_g = fft2(image[1].numpy())
        fft_b = fft2(image[2].numpy())

        fft_r_shifted = fftshift(fft_r)
        fft_g_shifted = fftshift(fft_g)
        fft_b_shifted = fftshift(fft_b)

        # Magnitude spectrum
        magnitude_r = np.log(1 + np.abs(fft_r_shifted))
        magnitude_g = np.log(1 + np.abs(fft_g_shifted))
        magnitude_b = np.log(1 + np.abs(fft_b_shifted))

        # Normalize to 0-255
        norm = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)) * 255
        magnitude_r = norm(magnitude_r)
        magnitude_g = norm(magnitude_g)
        magnitude_b = norm(magnitude_b)

        # Radial Entropy Calculation
        radial_entropy_r = self.radial_entropy(magnitude_r)
        radial_entropy_g = self.radial_entropy(magnitude_g)
        radial_entropy_b = self.radial_entropy(magnitude_b)
        radial_entropy_avg = (radial_entropy_r + radial_entropy_g + radial_entropy_b) / 3

        # Optional: Print or plot
        print(f"\033[1;36mRadial Entropy Avg:\033[0m {radial_entropy_avg[:20]}...")

        # Entropy calc
        entropy_r = self.calculate_entropy(magnitude_r)
        entropy_g = self.calculate_entropy(magnitude_g)
        entropy_b = self.calculate_entropy(magnitude_b)
        entropy_avg = (entropy_r + entropy_g + entropy_b) / 3
        #print(f"\033[1;33mAverage Entropy: {entropy_avg}\033[0m")

        # Combine channels
        magnitude_stack = np.stack([magnitude_r, magnitude_g, magnitude_b], axis=-1).astype(np.uint8)
        fourier_image_pil = Image.fromarray(magnitude_stack)

        # ---- Entropy Heatmap ----
        #print(image.shape)
        image_np = image.permute(1, 2, 0).numpy()  # [C, H,W] -> [H, W, C]
        image_np = (image_np * 255).astype(np.uint8)
        #print(image_np.min(), image_np.max(), image_np.dtype)

        # Split channels
        r = image_np[:, :, 0].astype(np.uint8)
        g = image_np[:, :, 1].astype(np.uint8)
        b = image_np[:, :, 2].astype(np.uint8)
        #print(f"{r.shape}, {g.shape}, {b.shape}")

        # Local entropy per channel
        entropy_r = local_entropy(r, disk(5))
        entropy_g = local_entropy(g, disk(5))
        entropy_b = local_entropy(b, disk(5))

        #Normalize
        entropy_r = norm(entropy_r)
        entropy_g = norm(entropy_g)
        entropy_b = norm(entropy_b)
        #print(f"{entropy_r.shape}, {entropy_g.shape}, {entropy_b.shape}")

        #stack back into RGB
        #entropy_rgb = np.stack([entropy_r, entropy_g, entropy_b], axis=-1).astype(np.uint8) # [H, W, C]
        #print(entropy_rgb.shape)
        colormap = cm.viridis
        entropy_avg_map = ((entropy_r + entropy_g + entropy_b) / 3).astype(np.uint8)
        colourmap = colormap(entropy_avg_map)
        colourmap = (colourmap[:, :, :3] * 255).astype(np.uint8)
        entropy_rgb_pil = Image.fromarray(colourmap)


        # Convert both images back to tensors
        fourier_tensor = ToTensor()(fourier_image_pil).unsqueeze(0).permute(0, 2, 3, 1)
        #heatmap_tensor = ToTensor()(heatmap_pil).unsqueeze(0).permute(0, 2, 3, 1)
        entropy_tensor = ToTensor()(entropy_rgb_pil).unsqueeze(0).permute(0, 2, 3, 1)

        return (fourier_tensor, entropy_tensor)

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
                "Text_1": ("STRING",{"default":""}),
                "CLIP_2": ("CLIP",),
                "Text_2": ("STRING",{"default":""}),
            }
        }



    def get_text_embeddings(self, clip, text):
        # Tokenize text
        tokens = clip.tokenize(text)
        embeddings = clip.encode_from_tokens_scheduled(tokens)

        # embeddings is a list of lists of tensors, we want the only entry in the the top level list (hence [0]) and the second entry of that subordinate list (hence [0][1])
        # but we need to reference the dictionary key to extract the element from that tensor to get the Pooled Attention
        pooled_embeddings = embeddings[0][1]['pooled_output']

        return pooled_embeddings

    def interrogation(self, Text_1, Text_2, CLIP_1, CLIP_2):
        #get text embeddings
        embedding1 = self.get_text_embeddings(CLIP_1, Text_1)
        embedding2 = self.get_text_embeddings(CLIP_2, Text_2)

        # Normalize the embeddings
        embedding1_norm = embedding1 / embedding1.norm(p=2)
        embedding2_norm = embedding2 / embedding2.norm(p=2)

        # Compute the cosine similarity as a dot product between the normalized vectors
        cos_sim = torch.mm(embedding1_norm, embedding2_norm.T)
        cos_sim_value = cos_sim.item()

        # Calculate distance between embeddings
        distance = torch.norm(embedding1 - embedding2, p=2)
        distance_value = distance.item()
        print(f'\033[1;33mText: "{Text_1}", "{Text_2}"\033[0m')
        print(f"\033[1;33mCosSim: {cos_sim_value}\033[0m")
        print(f"\033[1;33mEuclidean Distance: {distance_value}\033[0m")
        output_string = f"Cosine Similarity: {cos_sim_value}, Euclidean Distance: {distance_value}"

        return output_string

# Takes two images and outputs an image of the pixel-wise difference
class ImageDifference:
    FUNCTION = "ImageDiff"
    CATEGORY = "Strings&Things/Extras"
    DESCRIPTION = "Pairwise image comparrison"
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "contrast_boost": ("BOOLEAN", {
                    "default": False
                    }),
                "console_mse": ("BOOLEAN", {
                    "default": True
                    }),
            }
        }

    def ImageDiff(self, image1, image2, contrast_boost, console_mse):
        #print(image1.shape)
        #print(image2.shape)
        image1=image1.squeeze(0)
        image2=image2.squeeze(0)
        diff_image = torch.abs(image1 - image2) #Updated difference calculation to take absolute value; size is more important that directionality here. Would be nice to have a toggleable High Contrast Mode.
        #print(diff_image.shape)
        if contrast_boost:
            boost_factor = 5.0  # can make this a user input later
            diff_image = (diff_image * boost_factor).clamp(0,1)

        diff_image=diff_image.unsqueeze(0)
        #print(diff_image.shape)

        # Mean Squared Error
        if console_mse:
            mse = torch.mean((image1 - image2) ** 2)
            mse_percent = mse.item() * 100
            mse_percent_formatted = f"{mse_percent:.2f}%"
            print(f"\033[1;33mMean Square Error: {mse_percent_formatted}\033[0m")

        return diff_image,

'''
Part 3 MOSS-Image
    -MOSS to ComfyUI BBoxes
 '''

#Converts the Moss-Image Layout2Image json format into the correct
class MOSSToComfyUIBBoxes:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "moss_json": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": ""
                    }
                ),
                "source_width": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 1,
                        "max": 16384
                    }
                ),
                "source_height": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 1,
                        "max": 16384
                    }
                ),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("bboxes_json", "summary")

    OUTPUT_TOOLTIPS = (
        "Converted bounding-box elements in ComfyUI's 0-1000 coordinate format. Connect to the Create Bounding Boxes node's bboxes input.",
        "The summary section extracted from the prompt, can be connected to the MOSS-Image Layout prompt input to reproduce original image."
    )

    FUNCTION = "convert"
    CATEGORY = "MOSS/Utilities"

    def convert(self, moss_json, source_width, source_height):

        if not moss_json or not moss_json.strip():
            return ("", "{}")
        try:
            data = json.loads(moss_json)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid MOSS JSON: {exc}"
            ) from exc

        if not isinstance(data, dict):
            raise ValueError(
                "Expected a JSON object containing an items array."
            )

        if data.get("bbox_coord_space") != "abs_centralized":
            raise ValueError(
                "Expected bbox_coord_space='abs_centralized'."
            )

        if data.get("bbox_format", "yxyx") != "yxyx":
            raise ValueError(
                "Only bbox_format='yxyx' is currently supported."
            )

        items = data.get("items")

        if not isinstance(items, list):
            raise ValueError(
                "The MOSS JSON does not contain a valid items array."
            )

        converted = []

        for index, item in enumerate(items):

            if not isinstance(item, dict):
                raise ValueError(
                    f"Item {index} is not a JSON object."
                )

            bbox = item.get("bbox")

            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                raise ValueError(
                    f"Item {index} has an invalid bbox. "
                    "Expected [ymin, xmin, ymax, xmax]."
                )

            try:
                ymin, xmin, ymax, xmax = map(float, bbox)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Item {index} contains non-numeric bbox coordinates."
                ) from exc

            converted_bbox = [
                round((ymin + source_height / 2) / source_height * 1000),
                round((xmin + source_width / 2) / source_width * 1000),
                round((ymax + source_height / 2) / source_height * 1000),
                round((xmax + source_width / 2) / source_width * 1000),
            ]

            converted_bbox = [
                max(0, min(1000, value))
                for value in converted_bbox
            ]

            label = item.get("label", "object")

            element = {
                "type": "text" if label == "text" else "obj",
                "bbox": converted_bbox,
                "desc": item.get("caption", ""),
                "color_palette": item.get("palette", [])
            }

            if element["type"] == "text":
                element["text"] = item.get("text", "")

            converted.append(element)

        bboxes_json = json.dumps(
            converted,
            ensure_ascii=False,
            indent=4
        )

        # Extract and serialise the summary separately.
        summary = json.dumps(
            data.get("summary", {}),
            ensure_ascii=False,
            indent=4
        )

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