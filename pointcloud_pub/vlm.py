from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from PIL import Image
import requests
#import matplotlib.pyplot as plt
import time
import torch.nn.functional as F
import torch



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




  '''
  def visulize(self, mask):
      # shape: (num_prompts, H, W)
    e = time.time()
    # Plot
    fig, ax = plt.subplots(1, len(self.prompts) + 1, figsize=(3 * (len(self.prompts) + 1), 4))

    ax[0].imshow(self.image)
    ax[0].set_title("Image")
    ax[0].axis("off")

    for i, prompt in enumerate(self.prompts):
        ax[i + 1].imshow(mask[i], cmap="viridis")
        ax[i + 1].set_title(prompt)
        ax[i + 1].axis("off")

    plt.tight_layout()
    plt.show()
'''
