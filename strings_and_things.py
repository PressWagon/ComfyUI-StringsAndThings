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

# Takes up to 5 Lora names. Should now support chaining together multiple PWLoraNameCollector nodes.     
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

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Image",)
    FUNCTION = "FAnalysis"
    OUTPUT_NODE = True
    CATEGORY = "Strings&Things/Extras"

    def FAnalysis(self,image):
        # Rearrange the tensor from [batch_size, height, width, channels] to [batch_size, channels, height, width]
        image = image.permute(0, 3, 1, 2)
        
        # Dropping batch dimension
        image = image.squeeze(0)
        
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
        image = image.permute(0, 2, 3, 1) # Converting from [B, C, H, W] to [B, H, W, C]
        
        return (image,)
       
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
        diff_image = torch.abs(image1 - image2) #Updated difference calculation to take absolute value; size is more important that directionality here.
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
}        