"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import argparse
import torch

from src.model import DeepQNetwork
from src.env import ChromeDino
import cv2


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Chrome Dino""")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--fps", type=int, default=60, help="frames per second")
    parser.add_argument("--output", type=str, default="output/chrome_dino.mp4", help="the path to output video")

    args = parser.parse_args()
    return args


def test(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    model = DeepQNetwork()
    checkpoint_path = "{}/chrome_dino.pth".format(opt.saved_path)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    env = ChromeDino()
    state, raw_state, _, _ = env.step(0, True)
    state = torch.cat(tuple(state for _ in range(4)))[None, :, :, :]
    if torch.cuda.is_available():
        model.cuda()
        state = state.cuda()
    out = cv2.VideoWriter(opt.output, cv2.VideoWriter_fourcc(*"MJPG"), opt.fps, (600, 150))
    done = False
    while not done:
        prediction = model(state)[0]
        action = torch.argmax(prediction).item()
        next_state, raw_next_state, reward, done = env.step(action, True)
        out.write(raw_next_state)
        if torch.cuda.is_available():
            next_state = next_state.cuda()
        next_state = torch.cat((state[0, 1:, :, :], next_state))[None, :, :, :]
        state = next_state



if __name__ == "__main__":
    opt = get_args()
    test(opt)
