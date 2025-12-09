#!/usr/bin/python
# -*- coding: utf-8 -*-
from cv2.typing import MatLike
from models.network_fbcnn import FBCNN
from pathlib import Path
from tqdm import tqdm
from utils import utils_image as util
import argparse
import cv2
import numpy as np
import requests
import torch


def process_args() -> argparse.Namespace:
    """
    Process arguments for fbcnn.py.

    Returns:
        argparse.Namespace: Namespace containing all the specified arguments.
    """

    def percent(value):
        value = int(value)
        if 0 <= value <= 100:
            return value
        else:
            raise ValueError(
                f"Argument must be between 0 and 100. Got {value}."
            )

    parser = argparse.ArgumentParser(
        prog="fbcnn",
        description="Wrapper for the FBCNN JPEG de-artifacting tool",
    )

    parser.add_argument(
        "files",
        nargs="+",
        metavar="FILES",
    )

    parse_channels = parser.add_mutually_exclusive_group()
    parse_channels.add_argument(
        "--color",
        "--rgb",
        action="store_const",
        const=3,
        dest="channels",
    )
    parse_channels.add_argument(
        "--gray",
        "--grey",
        action="store_const",
        const=1,
        dest="channels",
    )

    parser.add_argument(
        "--qf",
        "-q",
        type=percent,
    )

    parser.set_defaults(channels=3)

    return parser.parse_args()


def main(args: argparse.Namespace):
    """
    Execution entry point for this script.
    """
    # Setup vars
    model_name, model_path = resolve_model(args)

    run_model(args, model_name, model_path)


def resolve_model(args: argparse.Namespace):
    """
    Gets the path to the model to use for the operation, downloading it if
    required.

    Args:
        args (argparse.Namespace): fbcnn.py arguments.

    Returns:
        (str, Path): [Model name, Path to model]
    """
    model_name = "fbcnn_color.pth" if args.channels == 3 else "fbcnn_gray.pth"
    model_path_local = Path("model_zoo", model_name)

    if model_path_local.is_file():
        # Model already exists.
        pass
    else:
        # Grab model from web.
        model_path_remote = f"https://github.com/jiaxi-jiang/FBCNN/releases/download/v1.0/{model_name}"

        print("Downloading model from ", model_path_remote)

        # Adapted from answer to 'Progress Bar while download file over http
        # with Requests' (https://stackoverflow.com/a/37573701), written by
        # leovp (https://stackoverflow.com/users/6223515/leovp). Licensed CC
        # BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/).
        r = requests.get(model_path_remote, allow_redirects=True, stream=True)
        rsize = int(r.headers.get("content-length", 0))

        # Enable loading bar
        with tqdm(total=rsize, unit="B", unit_scale=True) as pbar:
            # Write to file
            with open(model_path_local, "xb") as fh:
                for d in r.iter_content(chunk_size=1024):
                    pbar.update(len(d))
                    fh.write(d)

    return model_name, model_path_local


def create_model(model_path: Path, device: torch.device, n_channels: int):
    """
    Creates and instantiates the model.

    Args:
        model_path (Path): Path to the model.
        device (torch.device): The device to use (CPU or GPU).
        n_channels (int): Number of channels in the image.

    Returns:
        (FBCNN): The model created.
    """
    model = FBCNN(n_channels, n_channels)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()

    for _, v in model.named_parameters():
        v.requires_grad = False

    model = model.to(device)
    return model


def run_model(args: argparse.Namespace, model_name: str, model_path: Path):
    """
    Creates and runs the model.

    Args:
        args (argparse.Namespace): Arguments for fbcnn.py.
        model_name (str): Name of the model.
        model_path (Path): Path to the model.
    """
    print(f"Using model '{model_name}'")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_channels = int(args.channels)

    model = create_model(model_path, device, n_channels)

    jpg_paths = [Path(p) for p in args.files]
    for idx, jpg_path in enumerate(jpg_paths):
        png_path = jpg_path.with_suffix(".png")
        print("{:0>4d} ==> {:>16s}".format(idx + 1, jpg_path.name))

        # ------------------------------------
        # (1) img_L
        # ------------------------------------
        img_L = get_img_l(device, jpg_path, n_channels)

        # ------------------------------------
        # (2) img_E
        # ------------------------------------
        img_E = get_img_e(device, model, img_L, args.qf)

        util.imsave(img_E, png_path)


def get_img_l(device: torch.device, jpg_path: Path, n_channels: int):
    """
    Generates the imgL.

    Args:
        device (torch.device): The device to use (CPU or GPU).
        jpg_path (Path): Path to the JPEG image.
        n_channels (int): Number of channels in the image.

    Returns:
        (Tensor): The created Tensor image.
    """
    img_L = readimg(jpg_path, n_channels)

    img_L = util.uint2tensor4(img_L)
    img_L = img_L.to(device)
    return img_L


def get_img_e(
    device: torch.device,
    model: FBCNN,
    img_L: torch.Tensor,
    qf: int | None,
):
    """
    Generates the imgE.

    Args:
        device (torch.device): The device to use (CPU or GPU).
        model (FBCNN): Model used to de-noise the JPEG.
        img_L (torch.Tensor): Output of get_img_l().
        qf (int | None): The specified quality factor.

    Returns:
        uint8: The generated imgE.
    """
    if qf is None:
        img_E, qf_out = model(img_L)
    else:
        qf = int(qf)
        print("Flexible control by QF =", qf)

        qf_in = (
            torch.tensor([[1 - qf / 100]]).cuda()
            if device == torch.device("cuda")
            else torch.tensor([[1 - qf / 100]])
        )

        img_E, qf_out = model(img_L, qf_in)

    qf_out = 1 - qf_out
    img_E = util.tensor2single(img_E)
    img_E = util.single2uint(img_E)

    if qf is None:
        qf_predict = round(float(qf_out * 100))
        print(f"Predicted quality factor: {qf_predict}")

    return img_E


def readimg(path: Path, n_channels: int) -> MatLike:
    if n_channels == 1:
        img = cv2.imread(path.as_posix(), 0)  # cv2.IMREAD_GRAYSCALE

        if img:
            img = np.expand_dims(img, axis=2)  # HxWx1
        else:
            raise RuntimeError(f"cv2.imread has failed for '{path}'")
    elif n_channels == 3:
        img = cv2.imread(path.as_posix(), cv2.IMREAD_UNCHANGED)  # BGR or G

        if img is None:
            raise RuntimeError(f"cv2.imread has failed for '{path}'")
        elif img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # GGG
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
    else:
        raise ValueError(f"n_channels must be either 1 or 3. Got {n_channels}.")

    return img


if __name__ == "__main__":
    main(process_args())
