# SmartGamer

A Python framework that utilizes neural networks and computer vision to
train computers to autonomously play video games. Built using FastAI 
and OpenCV.

The AI learns by capturing your gameplay and categorizes the captured
images based on the provided inputs. For instance, whenever you press w
to walk forward, the AI will categorize the current game screen in the
"w" category. Once enough gameplay is captured, all images is fed into
a pretrained CNN (Convolutional Neural Network) called RESNET18. Using 
FastAI to tune the final layers of RESNET18 leads to a typical accuracy
of 95%, depending on the amount and quality of training data captured.

### Installation

This library currently only supports Windows out of the box. 
To use on MacOS/Linux systems, all win32gui functions needs to be replaced
/adapted to the relevant equivalent libraries on each system. No further
adjustments are needed.

Install the following libraries:

- win32gui
- pynput
- OpenCV (cv2)
- fastai
- PyTorch (CUDA 11.3 required, if NVIDIA GPU is available)
- pydirectinput

Simply import `Agent` from `Agent.py` and use accordingly (documentation
below). 

### Documentation

##### Agent

`def __init__ (game_inputs: list, game_name: str, none_input: bool = False, view_height: int = 300,
                 window_name_func: callable = None, view_width: int = 400, l_threshold: int = 130, u_threshold: int = 255)`

Initializes the Agent class.

- game_inputs: A list of input keys (as strings) the game uses. The program will only monitor/capture keys in this list
- game_name: Exact match of name of the game window. For non-exact matches, see window_name_func param below.
- none_input (default is False): Whether to capture case where no inputs are provided (no keys are pressed).
- view_width, view_height (default is 300, 400): Width and height of training images (not of actual game). Higher resolution leads to more details but slower processing speed.
- window_name_func (default is None): Optional custom function to match name of game window using game_name provided. For example, if game window name is non static but follows a pattern.
- l_threshold, u_threshold (default is 130, 255): Parameters used in Canny edge detection algorithm applied to each captured image. The bigger the difference between the two threshold, the more captured details (hence more processing time)

`def capture_game(path: str, fps_cap: int, pause_key: str, stop_key: str = None)`

Capture footage/images of the game window and separate them into categories based on
which key (or lack thereof) was pressed.

- path: Path of where image folders are stored
- fps_cap: The frequency or frames per second of the capture. I.e how many frames are captured each second.
- pause_key: Hotkey to pause/unpause capturing.
- stop_key (default is None): Hotkey to stop capturing and exit function.

`def train(img_path: str, epoch: int, batch_size: int = 40, model = models.resnet18,
              lr: float = 1.0e-02)`

Trains the neural network based on images captured.

- img_path: Path of where image folders are located
- epoch: How many cycles of training are required. Requires experimentation, higher isn't always better
- batch_size (default is 40): Batch size of images. Adjustments not recommended unless under specific circumstances
- model (default is resnet18): Option to adjust network model from available models offered by PyTorch. Other popular models include resnet32 etc.
- lr (default is 1e-02): Learning rate of the network. Slower learning rate means more epoch but more stable learning.

`def run(show_view: bool):`

Uses the trained model to autonomously play the given video game.

- show_view: Whether or not to display the "AI view" (what the network model sees). Will cause performance detriments, recommended for debugging purposes only.