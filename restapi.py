import argparse
import datetime
import io
import os
import json

from PIL import Image

import torch
from flask import Flask, request, Response, jsonify
from base64 import encodebytes

app = Flask(__name__)

DETECTION_URL = "/v1/object-detection/yolov5"
DATETIME_FORMAT = "%Y-%m-%d_%H-%M-%S-%f"


@app.route(DETECTION_URL, methods=["POST"])
def predict():
    if not request.method == "POST":
        return
    if request.files.get("image"):
        image_file = request.files["image"]
        image_bytes = image_file.read()
        img = Image.open(io.BytesIO(image_bytes))
        results = model([img], size=640)  # reduce size=320 for faster inference

        # Создание объекта Response
        response = Response()
        # Установка данных и заголовков
        response.headers['Content-Disposition'] = 'attachment; filename=data.json'
        response.headers['Content-Type'] = 'application/json'

        response.data = json.dumps(results.pandas().xyxy[0].to_json(orient="records"))

        # Сохраним изображение
        results.render()
        now_time = datetime.datetime.now().strftime(DATETIME_FORMAT)
        img_savename = f"detection_results/photos/{now_time}.png"
        Image.fromarray(results.ims[0]).save(img_savename)
        # Открытие файла изображения в двоичном режиме
        with open(img_savename, 'rb') as f:
            image_data = f.read()
        # Добавление файла изображения к объекту Response
        response.set_data(image_data)
        encoded_img = get_response_image(img_savename)
        AllData = {
            "Image": encoded_img,
            "Description": json.dumps(results.pandas().xyxy[0].to_json(orient="records"))
        }
        return jsonify(AllData)

def get_response_image(image_path):
    pil_img = Image.open(image_path, mode='r') # reads the PIL image
    byte_arr = io.BytesIO()
    pil_img.save(byte_arr, format='PNG') # convert the PIL image to byte array
    encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64
    return encoded_img

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask api exposing yolov5 model")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    #parser.add_argument('--model', default='yolov5s', help='model to run, i.e. --model yolov5s')
    args = parser.parse_args()

    model_path = 'best.pt'
    model = torch.hub.load('D:/Programming/Python_prog/Diploma/YOLOV5_2502/yolov5-master', 'custom', path=model_path, force_reload=True, source='local')
    model.conf = 0.50
    app.run(host="0.0.0.0", port=args.port)  # debug=True causes Restarting with stat
