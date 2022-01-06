from pynput import keyboard
from PIL import ImageGrab
import win32gui
import cv2
import numpy as np


def enum_windows():
    def callback(hwnd, results):
        results.append((hwnd, win32gui.GetWindowText(hwnd)))

    windows = []
    win32gui.EnumWindows(callback, windows)
    return windows


def capture_image(window_box, t1, t2, vw, vh):
    pil_image = ImageGrab.grab(window_box)
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    bw_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edge_img = cv2.Canny(bw_image, threshold1=t1, threshold2=t2)
    resized = cv2.resize(edge_img, (vw, vh))
    return resized

