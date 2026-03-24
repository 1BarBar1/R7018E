import torch
import torch.nn.functional as F
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from PIL import Image
import requests
#import matplotlib.pyplot as plt
import time





class Clipseg():
  def __init__(self, prompts=["Blocks"]):
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(self.device)

    self.processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    self.model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

    self.model.to(self.device)
    #.half() # Move to GPU and use FP16
    self.model.eval()

    self.prompts = prompts

  def get_segmentation(self,input):
    self.input = input[:, :, ::-1]
    self.input = Image.fromarray(self.input.astype("uint8"), mode="RGB")
    self.image = self.input
    vision_size = self.model.config.vision_config.image_size
    self.processor.image_processor.size = {
        "height": vision_size,
        "width": vision_size
    }
    self.processor.image_processor.do_resize = True
    self.processor.image_processor.do_center_crop = False
    inputs = self.processor(
        text=self.prompts,
        images=[self.input] * len(self.prompts),
        padding="max_length",
        return_tensors="pt"
        )

    #inputs = {k: v.to(self.device) for k,v in inputs.items()}
    # match model precision
    #inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
    # Predict
    with torch.no_grad():
        outputs = self.model(**inputs)
    logits = outputs.logits  # shape: (num_prompts, H, W)
    mask = torch.sigmoid(logits)
    mask = mask.unsqueeze(0).unsqueeze(0)  # NCHW
    mask = F.interpolate(mask, (logits.size(dim=0),480,640), mode="nearest")

    return mask[0,0].detach().cpu().numpy(), logits.cpu()




