from io import BytesIO

import PIL
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageOps
from flask import Flask, request, jsonify, send_file
from torchvision import transforms
from torchvision.ops import masks_to_boxes
from torchvision.utils import draw_bounding_boxes
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

app = Flask(__name__)

try:
    processor = CLIPSegProcessor.from_pretrained("./processor")
    model = CLIPSegForImageSegmentation.from_pretrained("./model")
    print('Found locally')
except:
    print('Retrieving model from remote')
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    processor.save_pretrained('./processor')
    model.save_pretrained('./model')


def scale_bbox(bbox, original_size, new_size):
    # bbox: [xmin, ymin, xmax, ymax]
    # original_size: [height, width]
    # new_size: [height, width]
    h_ratio = new_size[0] / original_size[0]
    w_ratio = new_size[1] / original_size[1]
    scaled_bbox = [
        int(bbox[0] * w_ratio),
        int(bbox[1] * h_ratio),
        int(bbox[2] * w_ratio),
        int(bbox[3] * h_ratio)
    ]
    return torch.tensor(scaled_bbox)


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def extract_bounding_boxes_from_segmentation_map(tensor, image, threshold):
    t = tensor

    t -= t.min()
    t /= t.max()
    to_convert = torch.unsqueeze(t > threshold, 0)

    boxes = masks_to_boxes(to_convert)

    transform = transforms.Compose([transforms.PILToTensor()])
    img = transform(image)  # .resize((352, 352)))

    scaled_box = scale_bbox(boxes[0], (352, 352), img.shape[1:])
    scaled_box = torch.unsqueeze(scaled_box, 0)

    return scaled_box


def tensor_to_bbox(tensor):
    xmin, ymin, xmax, ymax = tensor.numpy()[0].tolist()
    return {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}


def draw_bboxes(img, bboxes, labels):
    drawn_boxes = draw_bounding_boxes(img, torch.stack(bboxes), labels=labels, colors="red")
    return ImageOps.mirror(Image.fromarray(drawn_boxes.numpy().T).rotate(-90, PIL.Image.NEAREST, expand = 1))


@app.route('/draw', methods=['POST'])
def draw():
    data = request.json
    image_url = data.get('image_url')
    prompts = data.get('prompts')

    was_single = len(prompts) == 1

    if was_single:
        prompts = prompts * 2

    if not image_url or not prompts or len(prompts) == 0:
        return jsonify({'error': 'Invalid request payload'}), 400

    # Code for handling the POST request and producing the JSON output
    # Business logic goes here

    # getting the picture from the web
    try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status()  # Raises exception for non-2xx status codes
        image = Image.open(response.raw)
    except requests.exceptions.RequestException as e:
        return jsonify({'error': 'Error fetching image: {}'.format(str(e))}), 500
    except IOError as e:
        return jsonify({'error': 'Error opening image: {}'.format(str(e))}), 500

    inputs = processor(text=prompts, images=[image] * len(prompts), padding="max_length", return_tensors="pt")
    # predict
    with torch.no_grad():
        outputs = model(**inputs)
    preds = outputs.logits.unsqueeze(1)

    # building boxes

    bboxes = []

    for i in range(len(prompts)) if not was_single else range(1):
        label, threshold = prompts[i].split(':')

        bbox = extract_bounding_boxes_from_segmentation_map(preds[i][0], image, float(threshold))
        bboxes.append(bbox[0])

    transform = transforms.Compose([transforms.PILToTensor()])
    img = transform(image)
    print(bboxes)
    drawn_bbox = draw_bboxes(img, bboxes, prompts if not was_single else prompts[0])

    img_bytes = BytesIO()
    drawn_bbox.save(img_bytes, format='JPEG')
    img_bytes.seek(0)

    return send_file(img_bytes, mimetype='image/jpeg')


@app.route('/detect', methods=['POST'])
def predict():
    data = request.json
    image_url = data.get('image_url')
    prompts = data.get('prompts')

    was_single = len(prompts) == 1

    if was_single:
        prompts = prompts * 2

    if not image_url or not prompts or len(prompts) == 0:
        return jsonify({'error': 'Invalid request payload'}), 400

    # Code for handling the POST request and producing the JSON output
    # Business logic goes here

    # getting the picture from the web
    try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status()  # Raises exception for non-2xx status codes
        image = Image.open(response.raw)
    except requests.exceptions.RequestException as e:
        return jsonify({'error': 'Error fetching image: {}'.format(str(e))}), 500
    except IOError as e:
        return jsonify({'error': 'Error opening image: {}'.format(str(e))}), 500

    inputs = processor(text=prompts, images=[image] * len(prompts), padding="max_length", return_tensors="pt")
    # predict
    with torch.no_grad():
        outputs = model(**inputs)
    preds = outputs.logits.unsqueeze(1)

    # building boxes

    predictions = []

    for i in range(len(prompts)) if not was_single else range(1):
        label, threshold = prompts[i].split(':')
        print(preds.shape, preds[i].shape, preds[i][0].shape)
        bbox = extract_bounding_boxes_from_segmentation_map(preds[i][0], image, float(threshold))

        predictions.append({
            'label': label,
            'threshold': float(threshold),
            'bbox': tensor_to_bbox(bbox)
        })

    result = {
        'image_url': image_url,
        'predictions': predictions
    }

    return jsonify(result)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)



