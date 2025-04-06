import os
import time
import numpy as np
import torch
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from PIL import Image
import torchvision.transforms.functional as TF
from torch.autograd import Variable
from skimage import filters, segmentation, io
import ShuffleNetV2

# Get the absolute path to the application directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Ensure static/uploads directory exists
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Set Flask app configuration
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configure for Ngrok
app.config['PREFERRED_URL_SCHEME'] = 'https'

# Load CSV files with error handling
try:
    disease_info = pd.read_csv(os.path.join(BASE_DIR, "disease_info.csv"), encoding="utf-8-sig")
    supplement_info = pd.read_csv(os.path.join(BASE_DIR, "supplement_info.csv"), encoding="utf-8-sig")
except Exception as e:
    print("Error loading CSV files:", e)
    exit()

# Load model with error handling
try:
    model = ShuffleNetV2.ShuffleNetV2(39)
    model.load_state_dict(
        torch.load(os.path.join(BASE_DIR, "ShuffleNetV2_File_4.pt"), map_location=torch.device("cpu"))
    )
    model.eval()
except Exception as e:
    print("Error loading model:", e)
    exit()


def prediction(image_path):
    start_time = time.time()

    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image).unsqueeze(0)  # Ensure batch dimension

    input_data = (input_data - input_data.min()) / (input_data.max() - input_data.min())
    input_data = Variable(input_data, requires_grad=True)

    output = model(input_data)
    index = torch.argmax(output).item()

    # Generate heatmap
    model.zero_grad()
    one_hot_output = torch.zeros_like(output)
    one_hot_output[0, index] = 1
    output.backward(one_hot_output)

    gradients = input_data.grad.data.numpy()
    pooled_gradients = np.mean(gradients, axis=(0, 2, 3))
    activations = input_data.data.numpy()

    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]

    heatmap = np.mean(activations, axis=-1)
    heatmap = np.maximum(heatmap, 0)

    eps = 1e-8
    numer = heatmap - np.min(heatmap)
    denom = (heatmap.max() - heatmap.min()) + eps
    heatmap = numer / denom
    heatmap /= np.max(heatmap)

    heatmap_resized = cv2.resize(heatmap[0], (image.width, image.height))
    heatmap_rescaled = (heatmap_resized * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_rescaled, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)


    overlay_img = cv2.addWeighted(np.array(image), 0.5, heatmap_colored, 0.5, 0)

    # Save heatmap and overlay images
    heatmap_path = os.path.join(UPLOAD_FOLDER, "heatmap.jpg")
    overlay_path = os.path.join(UPLOAD_FOLDER, "overlay.jpg")
    cv2.imwrite(heatmap_path, heatmap_colored)
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))

    accuracy = round(torch.softmax(output, dim=1)[0][index].item() * 100, 2)
    duration = round(time.time() - start_time, 4)

    return index, os.path.basename(image_path), accuracy, duration, overlay_path


app = Flask(__name__)


@app.route("/")
def ai_engine_page():
    return render_template("index.html")


@app.route("/submit", methods=["POST"])
def submit():
    image = request.files["image"]
    filename = image.filename
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    image.save(file_path)

    pred, filename, accuracy, duration, overlay_path = prediction(file_path)

    title = disease_info["Disease Name"][pred]
    description = disease_info["Description"][pred]
    prevent = disease_info["Possible Steps"][pred]

    # Generate mask
    image_before = io.imread(file_path)
    gray_image = cv2.cvtColor(image_before, cv2.COLOR_BGR2GRAY)
    blurred = filters.gaussian(filters.sobel(gray_image), sigma=2.0)

    markers = np.zeros(blurred.shape, dtype=np.int32)
    ym, xm = blurred.shape[0] // 2, blurred.shape[1] // 2
    markers[0, :] = markers[-1, :] = markers[:, 0] = markers[:, -1] = 1
    markers[ym - 50 : ym + 50, xm - 20 : xm + 20] = 2

    mask = segmentation.watershed(blurred.astype(np.float32), markers)
    mask_path = os.path.join(UPLOAD_FOLDER, "mask.jpg")
    plt.imsave(mask_path, mask, cmap="gray")

    supplement_name = supplement_info["Supplement Name"][pred]
    supplement_image_url = supplement_info["Supplement Image"][pred]
    supplement_buy_link = supplement_info["Buy Link"][pred]
    drug_description = supplement_info["Drug Description"][pred]

    timestamp = int(time.time())
    
    # Convert absolute paths to relative URLs
    image_url = os.path.join("static", "uploads", filename)
    mask_url = os.path.join("static", "uploads", "mask.jpg")
    overlay_url = os.path.join("static", "uploads", "overlay.jpg")

    return render_template(
        "submit.html",
        title=title,
        desc=description,
        prevent=prevent,
        image_url=image_url,
        pred=pred,
        sname=supplement_name,
        simage=supplement_image_url,
        buy_link=supplement_buy_link,
        drug=drug_description,
        accuracy=accuracy,
        duration=duration,
        overlay_path=overlay_url,
        mask_path=mask_url,
        timestamp=timestamp
    )


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
