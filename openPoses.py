import os
import cv2
import base64
from PIL import Image
from io import BytesIO

def getPose(format, shot, number):
    path = f"./poses/{format}_{shot}/{number}.png"
    return getImageAtPath(path)

def getLewdPose(number):
    path = f"./lewdPoses/{number}.png"
    return getImageAtPath(path)

def getImageAtPath(path):
    if os.path.exists(path):
        print(f"Using pose at: {path}")
        binary_fc = open(path, 'rb').read()  # fc aka file_content
        base64_utf8_str = base64.b64encode(binary_fc).decode('utf-8')
        dataurl = f'data:image/png;base64,{base64_utf8_str}'
        return dataurl
    else:
        return None

def getBase64FromImage(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    b64 = base64.b64encode(buffered.getvalue())
    return b64.decode("utf-8")

def getImageFormBase64(b64):
    im_bytes = base64.b64decode(b64)   # im_bytes is a binary image
    im_file = BytesIO(im_bytes)  # convert image to file-like object
    img = Image.open(im_file)   # img is now PIL Image object
    return img

def getBase64StringFromOpenCV(openCVImage):
    _, im_arr = cv2.imencode('.jpg', openCVImage)  # im_arr: image in Numpy one-dim array format.
    im_bytes = im_arr.tobytes()
    im_b64 = base64.b64encode(im_bytes)
    return im_b64.decode("utf-8")
