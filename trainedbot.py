from fastai.vision.all import *
from PIL import ImageGrab
import win32gui
import cv2
import numpy as np
import os
import time
import pydirectinput


def label(path):
    return path.parent.name


network = load_learner('export.pkl')
print("Loaded model")


labels = ["down", "left", "none", "right", "up"]


def hwnd_callback(hwnd, result):
    if win32gui.GetWindowText(hwnd) == "Subway Surf":
        box = win32gui.GetWindowRect(hwnd)
        while True:
            start = time.time()
            pil_image = ImageGrab.grab(box)
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            bw_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edge_img = cv2.Canny(bw_image, threshold1=180, threshold2=255)
            resized = cv2.resize(edge_img, (400, 300))
            result = network.predict(resized)
            end = time.time()

            if result[0] != "none":
                pydirectinput.press(str(result[0]))
                time.sleep(0.1)

            print(f"Pred: {result[0]} Conf: {result[2][labels.index(result[0])]} Time: {end - start}")


win32gui.EnumWindows(hwnd_callback, [])