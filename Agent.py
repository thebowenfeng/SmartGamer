import win32gui
from pynput import keyboard
import time
import os
import cv2
from fastai.vision.all import *
import torch
import pydirectinput

from capture import enum_windows, capture_image


def label(path):
    return path.parent.name


class Agent:
    def __init__(self, game_inputs: list, game_name: str, none_input: bool = False, view_height: int = 300,
                 window_name_func: callable = None, view_width: int = 400, l_threshold: int = 130, u_threshold: int = 255):
        self.game_inputs = game_inputs
        self.game_name = game_name
        self.game_window = None
        self.capture_none = none_input
        self.vh = view_height
        self.vw = view_width
        self.l_threshold = l_threshold
        self.u_threshold = u_threshold

        # Keep track of several global states
        self.global_capture_state = False
        self.current_keys = []

        # Various settings
        self.pause_key = ""
        self.stop_key = None

        # Get game window
        all_windows = enum_windows()
        for hwnd, _ in all_windows:
            if window_name_func is not None:
                if window_name_func(win32gui.GetWindowText(hwnd)):
                    self.game_window = win32gui.GetWindowRect(hwnd)
                    break
            else:
                if self.game_name == win32gui.GetWindowText(hwnd):
                    self.game_window = win32gui.GetWindowRect(hwnd)
                    break

        if self.game_window is None:
            raise Exception("Game window not found")

        # Set up folders
        for key in game_inputs:
            if not os.path.isdir(key):
                os.mkdir(key)
        if none_input:
            if not os.path.isdir("none"):
                os.mkdir("none")

    def on_press(self, key):
        try:
            key_name = key.char
        except AttributeError:
            key_name = str(key).split(".")[1]

        if key_name == self.pause_key:
            self.global_capture_state = not self.global_capture_state
        else:
            if key_name not in self.current_keys and key_name in self.game_inputs:
                self.current_keys.append(key_name)

    def on_release(self, key):
        try:
            key_name = key.char
        except AttributeError:
            key_name = str(key).split(".")[1]

        self.current_keys = [k for k in self.current_keys if k != key_name]

    def capture_game(self, path: str, fps_cap: int, pause_key: str, stop_key: str = None):
        self.pause_key = pause_key
        self.stop_key = stop_key

        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()

        while True:
            if self.stop_key in self.current_keys:
                break

            if self.global_capture_state:
                image = capture_image(self.game_window, self.l_threshold, self.u_threshold, self.vw, self.vh)
                if len(self.current_keys) > 0:
                    for key in self.game_inputs:
                        if key in self.current_keys:
                            print(key)
                            cv2.imwrite(path + '/' + key + '/' + str(os.listdir(key).__len__()) + '.png', image)

                else:
                    if self.capture_none:
                        print("No key pressed")
                        cv2.imwrite(path + '/' + "none" + '/' + str(os.listdir("none").__len__()) + '.png', image)
            else:
                print("Paused")

            time.sleep(1 / fps_cap)

        listener.stop()

    def train(self, img_path: str, epoch: int, batch_size: int = 40, model = models.resnet18,
              lr: float = 1.0e-02):
        images = get_image_files(img_path)
        data = ImageDataLoaders.from_path_func(img_path, images, label, bs=batch_size, num_workers=0)
        learn = cnn_learner(data, model, metrics=error_rate)
        learn.fine_tune(epoch, base_lr=lr)
        learn.export()

    def run(self, show_view: bool):
        network = load_learner("export.pkl")
        print("Model loaded")
        input_list = self.game_inputs
        if self.capture_none:
            input_list.append("none")

        while True:
            start = time.time()
            image = capture_image(self.game_window, self.l_threshold, self.u_threshold, self.vw, self.vh)
            if show_view:
                cv2.imshow('AIView', image)
                cv2.waitKey(1)

            prediction = network.predict(image)
            end = time.time()

            if prediction[0] != "none":
                pydirectinput.press(str(prediction[0]))
                time.sleep(1)

            print(f"Pred: {prediction[0]} Conf: {prediction[2][input_list.index(prediction[0])]} Time: {end - start}")




test = Agent(["left", "right", "up", "down"], "Subway Surf", none_input=True)
#test.capture_game(1000, "p")
#test.train(os.getcwd(), 6)
test.run(True)
