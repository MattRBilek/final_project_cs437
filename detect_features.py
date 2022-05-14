import cv2
from enum import Enum
import numpy as np
import math
# some code for face detection pulled from https://github.com/shantnu/FaceDetect/blob/master/face_detect.py
# namely this casc file
cascPath = "casc.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

class FEATURES(Enum):
    HAIR = 0
    SHIRT = 1
    HEIGHT = 2

hair_to_color = [[88, 75, 66], # BLACK
                  [144,84,47], # BROWN
                  [251,231,161], # BLOND
                  [176, 101, 0] # GINGER
                  ]
hair_to_color = np.array(hair_to_color)

class HAIR_COLORS(Enum):
    BLACK = 0
    BROWN = 1
    BLOND = 2
    GINGER = 3

def hair_to_string(num):
    if num is None:
        return None
    if num == HAIR_COLORS.BLACK.value:
        return "black"
    elif num == HAIR_COLORS.BROWN.value:
        return "brown"
    elif num == HAIR_COLORS.BLOND.value:
        return "blond"
    else:
        return "ginger"

shirt_to_color = [[255, 165, 0], # ORANGE
                  [32,42,180], # BLUE
                  [49, 45, 43], # BLACK
                  [180, 180, 180]  #WHITE
                  ]
shirt_to_color = np.array(shirt_to_color)


class SHIRT_COLORS(Enum):
    ORANGE = 0
    BLUE = 1
    BLACK = 2
    WHITE = 3

def shirt_to_string(num):
    if num is None:
        return None

    if num == SHIRT_COLORS.BLACK.value:
        return "black"
    elif num == SHIRT_COLORS.BLUE.value:
        return "blue"
    elif num == SHIRT_COLORS.WHITE.value:
        return "white"
    else:
        return "orange"


box_size = 1
pixel_to_height = None # inch per pixel

def calibrate_height(filename, viz=False): # https://docs.opencv.org/4.x/dc/d0d/tutorial_py_features_harris.html
    global pixel_to_height
    img = cv2.imread(filename)
    img = img[300:400,550:700]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    dst = cv2.dilate(dst, None)

    dist_perect = .25

    img[dst > dist_perect * dst.max()] = [0, 0, 255]
    points = dst > dist_perect * dst.max()

    points = np.where(points)
    points = np.array(points).T
    for point in points:
        for temp in points:
            if np.all(temp != point):
                dist = np.abs(temp - point)
                if dist[1] / (dist[0] + dist[1]) > .95:
                    if pixel_to_height is None or pixel_to_height < box_size / dist[1]:
                        pixel_to_height = box_size / dist[1]

    if viz:
        print(pixel_to_height)
        cv2.imshow('dst', img)
        cv2.waitKey(0)

def get_height_bootstrap(face, image, viz=False):
    pre_cal_pix_to_inch = 0.142     # got with above
    pre_cal_bot = 670   # found though testing
    x, y, width, height = face
    height *= .3
    height = int(height)
    y -= height
    height = pre_cal_bot - y

    if viz:
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.imshow("Faces found", image)
        print(height * pre_cal_pix_to_inch)
        cv2.waitKey(0)

    return height * pre_cal_pix_to_inch

#this would work however it requires a higher quality camera that we didnt want to buy
def old_get_height(image, viz=False):
    image = image[:,300:800]
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    (regions, _) = hog.detectMultiScale(image,
                                    winStride=(4, 4),
                                    scale=1.0)
    if len(regions) == 0:
        return None

    x, y, w, h = regions[0]
    print(len(regions))
    if viz:
        cv2.rectangle(image, (x, y),
                      (x + w, y + h),
                      (0, 0, 255), 2)
        cv2.imshow("Image", image)
        cv2.waitKey(0)
    print(h * pixel_to_height)
    return h * pixel_to_height

def get_features(image, viz=False):
    faces = get_faces(image)
    if len(faces) == 0:
        return None
    features = {}
    features['hair'] = hair_to_string(get_hair(faces[0],image, viz))
    features['shirt'] = shirt_to_string(get_shirt(faces[0],image, viz))
    features['height'] = get_height_bootstrap(faces[0], image, viz)
    print(features)
    return features

def get_hair(face, image, viz = False):
    x, y, width, height = face
    height *= .3
    height = int(height)
    y -= height
    hair_classes = [0] * len(hair_to_color)

    box = image[y:y+height, x:x+width]
    h = image.shape[0]
    w = image.shape[1]
    for hair in HAIR_COLORS:
        color = hair_to_color[hair.value]
        hair_classes[hair.value] = get_box_count(box,color)

    if viz:
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.imshow("Faces found", image)
        print(hair_classes)
        print(np.array(hair_classes).argmax())
        cv2.waitKey(0)

    return np.array(hair_classes).argmax()



def get_shirt(face, image, viz=False):
    x, y, width, height = face
    y += height
    y += int(height*.3)
    shirt_classes = [0] * len(shirt_to_color)

    box = image[y:y + height, x:x + width]
    h = image.shape[0]
    w = image.shape[1]
    for shirt in SHIRT_COLORS:
        color = shirt_to_color[shirt.value]
        shirt_classes[shirt.value] = get_box_count(box,color)



    if viz:
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.imshow("Faces found", image)
        print(shirt_classes)
        print(np.array(shirt_classes).argmax())
        cv2.waitKey(0)
    return np.array(shirt_classes).argmax()


def get_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )
    return faces


def get_box_count(box,color, threshold=np.array([30,30,30])):
    temp = np.abs(box-color)
    return np.all(temp < threshold, axis=2).sum()



if __name__ == "__main__":
    #calibrate_height(cv2.imread(u"corn.jpg"), True)
    get_features(cv2.imread(u"test3.jpg"), True)
