# Controlled text-to-image generation with ControlNet on Inference Endpoints

> ## Excerpt
> Learn how to deploy ControlNet Stable Diffusion Pipeline on Hugging Face Inference Endpoints to generate controlled images.

---
ControlNet is a neural network structure to control diffusion models by adding extra conditions.

With [ControlNet](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/controlnet), users can easily condition the generation with different spatial contexts such as a depth map, a segmentation map, a scribble, keypoints, and so on!

We can turn a cartoon drawing into a realistic photo with incredible coherence.

![example](https://www.philschmid.de/_next/image?url=%2Fstatic%2Fblog%2Fstable-diffusion-controlnet-endpoint%2Fexample.jpg&w=2048&q=75)

Suppose you are now as impressed as I am. In that case, you are probably asking yourself: “ok, how can I integrate ControlNet into my applications in a scalable, reliable, and secure way? How can I use it as an API?”.

That's where Hugging Face Inference Endpoints can help you! [🤗 Inference Endpoints](https://huggingface.co/inference-endpoints) offers a secure production solution to easily deploy Machine Learning models on dedicated and autoscaling infrastructure managed by Hugging Face.

This blog post will teach you how to create ControlNet pipelines with Inference Endpoints using the [custom handler](https://huggingface.co/docs/inference-endpoints/guides/custom_handler) feature. [Custom handlers](https://huggingface.co/docs/inference-endpoints/guides/custom_handler) allow users to modify, customize and extend the inference step of your model.

![architecture](data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7)

Before we can get started, make sure you meet all of the following requirements:

1.  An Organization/User with an active credit card. (Add billing [here](https://huggingface.co/settings/billing))
2.  You can access the UI at: [https://ui.endpoints.huggingface.co](https://ui.endpoints.huggingface.co/endpoints)

The Tutorial will cover how to:

1.  [Create ControlNet Inference Handler](https://www.philschmid.de/stable-diffusion-controlnet-endpoint#1-create-controlnet-inference-handler)
2.  [Deploy Stable Diffusion ControlNet pipeline as Inference Endpoint](https://www.philschmid.de/stable-diffusion-controlnet-endpoint#2-deploy-stable-diffusion-controlnet-pipeline-as-inference-endpoint)
3.  [Integrate ControlNet as API and send HTTP requests using Python](https://www.philschmid.de/stable-diffusion-controlnet-endpoint#3-integrate-controlnet-as-api-and-send-http-requests-using-python)

### TL;DR;

You can directly hit “deploy” on this repository to get started: [https://huggingface.co/philschmid/ControlNet-endpoint](https://huggingface.co/philschmid/ControlNet-endpoint)

## 1\. Create ControlNet Inference Handler

This tutorial is not covering how you create the custom handler for inference. If you want to learn how to create a custom Handler for Inference Endpoints, you can either checkout the [documentation](https://huggingface.co/docs/inference-endpoints/guides/custom_handler) or go through [“Custom Inference with Hugging Face Inference Endpoints”](https://www.philschmid.de/custom-inference-handler)

We are going to deploy [philschmid/ControlNet-endpoint](https://huggingface.co/philschmid/ControlNet-endpoint), which implements the following `handler.py` for [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5). The repository is not including the weights and loads the model on endpoint creation. This means you can easily adjust which Stable Diffusion model you want to use by editing the `id` in the [handler](https://huggingface.co/philschmid/ControlNet-endpoint/blob/9fbec2fdc74198b987863895a27bc47619dacc83/handler.py#L64).

The custom handler implements a `CONTROLNET_MAPPING`, allowing us to define different control types on inference type. Supported control types are `canny_edge`, `pose`, `depth`, `scribble`, `segmentation`, `normal`, `hed`, and `though`.

The handler expects the following payload.

```
{
  "inputs": "A prompt used for image generation",
  "negative_prompt": "low res, bad anatomy, worst quality, low quality",
  "controlnet_type": "depth",
  "image": "iVBORw0KGgoAAAANSUhEUgAAAgAAAAIACAIAAAB7GkOtAAAABGdBTUEAALGPC"
}
```

The `image` attribute includes the image as `base64` string. You can additionally provide [hyperparameters](https://huggingface.co/philschmid/ControlNet-endpoint/blob/9fbec2fdc74198b987863895a27bc47619dacc83/handler.py#L94) to customize the pipeline, including `num_inference_steps`, `guidance_scale`, `height` , `width`, and `controlnet_conditioning_scale`.

## 2\. Deploy Stable Diffusion ControlNet pipeline as Inference Endpoint

UI: [https://ui.endpoints.huggingface.co/new?repository=philschmid/ControlNet-endpoint](https://ui.endpoints.huggingface.co/new?repository=philschmid/ControlNet-endpoint)

We can now deploy the model as an Inference Endpoint. We can deploy our custom Custom Handler the same way as a regular Inference Endpoint.

Select the repository, the cloud, and the region, adjust the instance and security settings, and deploy.

![repository](data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7)

Since the weights are not included in the repository, the UI suggests a CPU instance to deploy the model.

We want to change the instance to `GPU [medium] · 1x Nvidia A10G` to get decent performance for our pipeline.

![instance](data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7)

We can then deploy our model by clicking “Create Endpoint”

## 3\. Integrate ControlNet as API and send HTTP requests using Python

We are going to use `requests` to send our requests. (make your you have it installed `pip install requests`). We need to replace the `ENDPOINT_URL` and `HF_TOKEN` with our values and then can send a request. Since we are using it as an API, we need to provide at least a `prompt` and `image`.

To test the API, we download a sample image from the repository

```
wget https://huggingface.co/philschmid/ControlNet-endpoint/blob/main/huggingface.png
```

We can now run our python script using the `huggingface.png` to edit the image.

```
import json
from typing import List
import requests as r
import base64
from PIL import Image
from io import BytesIO

ENDPOINT_URL = "" # your endpoint url
HF_TOKEN = "" # your huggingface token `hf_xxx`

# helper image utils
def encode_image(image_path):
  with open(image_path, "rb") as i:
    b64 = base64.b64encode(i.read())
  return b64.decode("utf-8")

def predict(prompt, image, negative_prompt=None, controlnet_type = "normal"):
    image = encode_image(image)

    # prepare sample payload
    request = {"inputs": prompt, "image": image, "negative_prompt": negative_prompt, "controlnet_type": controlnet_type}
    # headers
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
        "Accept": "image/png" # important to get an image back
    }

    response = r.post(ENDPOINT_URL, headers=headers, json=request)
    if response.status_code != 200:
        print(response.text)
        raise Exception("Prediction failed")
    img = Image.open(BytesIO(response.content))
    return img

prediction = predict(
  prompt = "cloudy sky background lush landscape house and green trees, RAW photo (high detailed skin:1.2), 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3",
  negative_prompt ="lowres, bad anatomy, worst quality, low quality, city, traffic",
  controlnet_type = "hed",
  image = "huggingface.png"
)

prediction.save("result.png")
```

The result of the request should be a `PIL` image we can display:

## Conclusion

We successfully created and deployed a ControlNet Stable Diffusion inference handler to Hugging Face Inference Endpoints in less than 30 minutes.

Having scalable, secure API Endpoints will allow you to move from the experimenting (space) to integrated production workloads, e.g., Javascript Frontend/Desktop App and API Backend.

Now, it's your turn! [Sign up](https://ui.endpoints.huggingface.co/new) and create your custom handler within a few minutes!

___

Thanks for reading! If you have any questions, feel free to contact me on [Twitter](https://twitter.com/_philschmid) or [LinkedIn](https://www.linkedin.com/in/philipp-schmid-a6a2bb196/).
