# ComfyUI Strings&Things

A collection of ComfyUI custom nodes for formatting and debugging string data with the intention of collecting generation data to be processed by a custom node pack like comfy-image-saver, as well as miscellaneous extra nodes to experiment with.

# Part 1 - String Formatting
Lora Selector - lets you pick a LoRa while also outputting its name as a string

Lora Name Collector - takes up to 5 LoRas. Does not currently stack with itself (will fix this eventually)

Debug String - Attempts to print input to the console. Primarily useful for strings, but tries to take anything as input.

Formatting (Pre and Post text) and Concattenating 3 Strings - Added pre and post text to up to 3 input strings and concatenates them in order with a separator. Empty body inputs will not have teir pre and post text included. 

Formatting Single String - Pre and post text for a single string.

# Part 2 - Extras
Mosaic Effect Node - Applies a mosaic effect (aka tile censoring) to the entire image. Useful for pixelart style generations, or in combination with BBox/SEGS for censoring. (Will add example workflows at some point)

Fourier Analysis Node - Some people claim some stuff about AI and repetitive patterns being visible in the fourier analysis. Don't ask me, I'm not a scientist, I just think they look cool. 

Text Embedding Interrogator (Cosine Similarity and Euclidean Distance calculator) - Calculates the Cosine Similarity and Euclidean Distance between the Pooled Attention CLIP text embeddings.
