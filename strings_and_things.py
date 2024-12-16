import folder_paths

import torch
from torchvision.transforms import ToPILImage, ToTensor
from scipy.fftpack import fft2, fftshift
from PIL import Image
import numpy as np

class AnyType(str):
    """A special type that can be connected to any other types. Credit to pythongosssss"""

    def __ne__(self, __value: object) -> bool:
        return False


any_type = AnyType("*")

# Picks a Lora as an output to both send to a lora loader and the metadata as a string
class LoraSelector:
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

# Takes up to 5 Lora names. Is not setup for chaining multiple LoraNameCollector nodes together (TO DO: fix this, chaining creates multiple lists)        
class LoraNameCollector:
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
    
    def LoraNameConcat(Lora_A=None, Lora_B=None, Lora_C=None, Lora_D=None, Lora_E=None):
        # Filter out None values and store the remaining strings in a list
        lora_names = list(filter(None, [Lora_A, Lora_B, Lora_C, Lora_D, Lora_E]))
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
        image = image_pil = ToPILImage()(image[0])  

        # Get image dimensions
        width, height = image_pil.size

        # Create a new image for the output (same size)
        mosaic_image = Image.new("RGB", (width, height))

        # Process the image in tiles
        for y in range(0, height, tile_size):
            for x in range(0, width, tile_size):
                # Define the tile box
                box = (x, y, x + tile_size, y + tile_size)

                # Crop the tile from the image
                tile = image_pil.crop(box)

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

    def get_median_color(self, tile):
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

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Image",)
    FUNCTION = "FAnalysis"
    OUTPUT_NODE = True
    CATEGORY = "Strings&Things/Extras"

    @staticmethod
    def FAnalysis(self, image):

        # Rearrange the tensor from [batch_size, height, width, channels] to [batch_size, channels, height, width]
        image = image.permute(0, 3, 1, 2)
        
        #dropping batch dimension
        image = (image).squeeze(0)
        
        # Perform Fourier Transform on each channel separately
        fft_r = fft2(image[0].numpy())  # Red channel
        fft_g = fft2(image[1].numpy())  # Green channel
        fft_b = fft2(image[2].numpy())  # Blue channel

        # Shift the zero frequency component to the center
        fft_r_shifted = fftshift(fft_r)
        fft_g_shifted = fftshift(fft_g)
        fft_b_shifted = fftshift(fft_b)

        # Take log of the magnitude for visualization and normalize to 0-255 range
        magnitude_r = np.log(1 + np.abs(fft_r_shifted))
        magnitude_g = np.log(1 + np.abs(fft_g_shifted))
        magnitude_b = np.log(1 + np.abs(fft_b_shifted))

        # Normalize each channel to range [0, 255]
        magnitude_r = (magnitude_r - np.min(magnitude_r)) / (np.max(magnitude_r) - np.min(magnitude_r)) * 255
        magnitude_g = (magnitude_g - np.min(magnitude_g)) / (np.max(magnitude_g) - np.min(magnitude_g)) * 255
        magnitude_b = (magnitude_b - np.min(magnitude_b)) / (np.max(magnitude_b) - np.min(magnitude_b)) * 255

        # Convert the magnitude to uint8 type
        magnitude_r = magnitude_r.astype(np.uint8)
        magnitude_g = magnitude_g.astype(np.uint8)
        magnitude_b = magnitude_b.astype(np.uint8)

        # Stack the channels back into a 3D array (height, width, 3)
        magnitude_image = np.stack([magnitude_r, magnitude_g, magnitude_b], axis=-1)

        # Convert the magnitude image back to a PIL image
        fourier_image_pil = Image.fromarray(magnitude_image)

        # Convert back to tensor
        image = ToTensor()(fourier_image_pil).unsqueeze(0)  # Add batch dimension back
        image = image.permute(0, 2, 3, 1) #converting from [B, C, H, W] to [B, H, W, C]
        
        return (image,)
        
        
NODE_CLASS_MAPPINGS = {
    "LoraSelector": LoraSelector,
    "LoraNameCollector": LoraNameCollector,
    "DebugString": DebugString,
    "FormatConcatStrings": FormatConcatStrings,
    "FormattingSingle": FormattingSingle,
    "MosaicEffectNode": MosaicEffectNode,
    "FourierAnalysisNode": FourierAnalysisNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoraSelector": "Lora Selector",
    "LoraNameCollector": "Lora Name Collector",
    "DebugString": "🔧 Debug String",
    "FormatConcatStrings": "Formatting and Concatenating Strings",
    "FormattingSingle": "Formatting Single String",
    "MosaicEffectNode": "Apply Mosaic Effect",
    "FourierAnalysisNode": "Fourier Analysis",
}        