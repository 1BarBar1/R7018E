# Run this on your LAPTOP
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

# This downloads it from the internet
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

# This saves it to a folder on your laptop
processor.save_pretrained("./clipseg-local")
model.save_pretrained("./clipseg-local")