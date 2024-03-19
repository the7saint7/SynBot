import os
import base64

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
