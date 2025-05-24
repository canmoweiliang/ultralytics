from ultralytics import YOLO
import requests
from PIL import Image
from io import BytesIO
import os
import re

url = "https://api.csdi.gov.hk/apim/dataquery/api/?id=td_rcd_1638952287148_39267&layer=traffic_camera_locations_en&limit=10&offset=0"
response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    dir_name = re.sub(r'\D', '', data['timeStamp'])
    dir_path = './runs/{}'.format(dir_name)
    pictures = []
    locations = []
    for feature in data['features']:
        pic_url = feature['properties']['url']
        req = requests.get(pic_url)
        picture = Image.open(BytesIO(req.content))
        picture = picture.resize((picture.width * 2, picture.height * 2), Image.BICUBIC)
        pictures.append(picture)
        locations.append(feature['properties']['description'])
    model = YOLO("./runs/detect/train23/weights/best.pt")
    results = model.predict(source=pictures, imgsz=640)

    os.makedirs(dir_path, exist_ok=True)

    for location, result in zip(locations, results):
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        filename = os.path.join(dir_path, "{}.jpg".format(location))
        result.save(filename=filename)  # save to disk
else:
    print('请求失败，状态码:', response.status_code)
