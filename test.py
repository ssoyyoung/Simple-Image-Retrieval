from yolov3.pirs_utils_v2 import *
from PIL import Image

path = "testImg/test.jpg"

#img0 = cv2.imread(path)
#image = Image.open("testImg/test.jpg").convert('RGB')
#img, img0 = transfer_b64(image, mode="square")

img, img0 = transfer(path, mode="square")
cv2.imwrite('check.jpg', img)


print(img.shape)
print(img0.shape)


def letterbox(img, new_shape=416, color=(128,128,128), mode='square'):
    # Resize a rectangular image to a 32 pixel multiple rectangle
    shape = img.size  # current shape [height, width]

    if isinstance(new_shape, int):
        ratio = float(new_shape) / max(shape)
    else:
        ratio = max(new_shape) / max(shape)  # ratio  = new / old

    ratiow, ratioh = ratio, ratio
    new_unpad = (int(round(shape[0] * ratio)), int(round(shape[1] * ratio)))

    # Compute padding https://github.com/ultralytics/yolov3/issues/232
    dw = (new_shape - new_unpad[0]) / 2  # width padding
    dh = (new_shape - new_unpad[1]) / 2  # height padding

    if shape[::-1] != new_unpad:  # resize
        if ratio < 1:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_AREA)  # INTER_AREA is better, INTER_LINEAR is faster
        else:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LANCZOS4)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratiow, ratioh, dw, dh