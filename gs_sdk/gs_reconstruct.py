import time
import math
import os

import cv2
import numpy as np
from scipy import fftpack
import torch
import torch.nn as nn
import torch.nn.functional as F


class BGRXYMLPNet(nn.Module):
    """
    The Neural Network architecture for GelSight calibration.
    """

    def __init__(self):
        super(BGRXYMLPNet, self).__init__()
        input_size = 5
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 2)

    def forward(self, x):
        x = F.relu_(self.fc1(x))
        x = F.relu_(self.fc2(x))
        x = F.relu_(self.fc3(x))
        x = self.fc4(x)
        return x


class Reconstructor:
    """
    The GelSight reconstruction class.

    This class handles 3D reconstruction from calibrated GelSight images.
    """

    def __init__(self, model_path, contact_mode="standard", device="cpu"):
        """
        Initialize the reconstruction model.
        Contact mode "flat" means the object in contact is flat, so a different threshold
        is used to determine contact mask.

        :param model_path: str; the path of the calibrated neural network model.
        :param contact_mode: str {"standard", "flat"}; the mode to get the contact mask.
        :param device: str {"cuda", "cpu"}; the device to run the model.
        """
        self.model_path = model_path
        self.contact_mode = contact_mode
        self.device = device
        self.bg_image = None
        # Load the gxy model
        if not os.path.isfile(model_path):
            raise ValueError("Error opening %s, file does not exist" % model_path)
        self.gxy_net = BGRXYMLPNet()
        self.gxy_net.load_state_dict(torch.load(model_path), self.device)
        self.gxy_net.eval()

    def load_bg(self, bg_image):
        """
        Load the background image.

        :param bg_image: np.array (H, W, 3); the background image.
        """
        self.bg_image = bg_image

        # Calculate the gradients of the background
        bgrxys = image2bgrxys(bg_image).reshape(-1, 5)
        features = torch.from_numpy(bgrxys).float().to(self.device)
        with torch.no_grad():
            gxyangles = self.gxy_net(features)
            gxyangles = gxyangles.cpu().detach().numpy()
            self.bg_G = np.tan(
                gxyangles.reshape(bg_image.shape[0], bg_image.shape[1], 2)
            )

    def get_surface_info(
        self, image, ppmm, color_dist_threshold=15, height_threshold=0.2
    ):
        """
        Get the surface information including gradients (G), height map (H), and contact mask (C).

        :param image: np.array (H, W, 3); the gelsight image.
        :param ppmm: float; the pixel per mm.
        :param color_dist_threshold: float; the color distance threshold for contact mask.
        :param height_threshold: float; the height threshold for contact mask.
        :return G: np.array (H, W, 2); the gradients.
                H: np.array (H, W); the height map.
                C: np.array (H, W); the contact mask.
        """
        # Calculate the gradients
        bgrxys = image2bgrxys(image).reshape(-1, 5)
        features = torch.from_numpy(bgrxys).float().to(self.device)
        with torch.no_grad():
            gxyangles = self.gxy_net(features)
            gxyangles = gxyangles.cpu().detach().numpy()
            G = np.tan(gxyangles.reshape(image.shape[0], image.shape[1], 2))
            if self.bg_image is not None:
                G = G - self.bg_G
            else:
                raise ValueError("Background image is not loaded.")

        # Calculate the height map
        H = poisson_dct_neumann(G[:, :, 0], G[:, :, 1]).astype(np.float32)

        # Calculate the contact mask
        if self.contact_mode == "standard":
            # Filter by color difference
            diff_image = image.astype(np.float32) - self.bg_image.astype(np.float32)
            color_mask = np.linalg.norm(diff_image, axis=-1) > color_dist_threshold
            color_mask = cv2.dilate(
                color_mask.astype(np.uint8), np.ones((7, 7), np.uint8)
            )
            color_mask = cv2.erode(
                color_mask.astype(np.uint8), np.ones((15, 15), np.uint8)
            )

            # Filter by height
            cutoff = np.percentile(H, 85) - height_threshold / ppmm
            height_mask = H < cutoff

            # Combine the masks
            C = np.logical_and(color_mask, height_mask)
        elif self.contact_mode == "flat":
            # Find the contact mask based on color difference
            diff_image = image.astype(np.float32) - self.bg_image.astype(np.float32)
            color_mask = np.linalg.norm(diff_image, axis=-1) > 10
            color_mask = cv2.dilate(
                color_mask.astype(np.uint8), np.ones((15, 15), np.uint8)
            )
            C = cv2.erode(
                color_mask.astype(np.uint8), np.ones((25, 25), np.uint8)
            ).astype(np.bool_)
        return G, H, C


def image2bgrxys(image):
    """
    Convert a bgr image to bgrxy feature.

    :param image: np.array (H, W, 3); the bgr image.
    :return: np.array (H, W, 5); the bgrxy feature.
    """
    ys = np.linspace(0, 1, image.shape[0], endpoint=False, dtype=np.float32)
    xs = np.linspace(0, 1, image.shape[1], endpoint=False, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    bgrxys = np.concatenate(
        [image.astype(np.float32) / 255, xx[..., np.newaxis], yy[..., np.newaxis]],
        axis=2,
    )
    return bgrxys


def poisson_dct_neumann(gx, gy):
    """
    2D integration of depth from gx, gy using Poisson solver.

    :param gx: np.array (H, W); the x gradient.
    :param gy: np.array (H, W); the y gradient.
    :return: np.array (H, W); the depth map.
    """
    # Compute Laplacian
    gxx = 1 * (
        gx[:, (list(range(1, gx.shape[1])) + [gx.shape[1] - 1])]
        - gx[:, ([0] + list(range(gx.shape[1] - 1)))]
    )
    gyy = 1 * (
        gy[(list(range(1, gx.shape[0])) + [gx.shape[0] - 1]), :]
        - gy[([0] + list(range(gx.shape[0] - 1))), :]
    )
    f = gxx + gyy

    # Right hand side of the boundary condition
    b = np.zeros(gx.shape)
    b[0, 1:-2] = -gy[0, 1:-2]
    b[-1, 1:-2] = gy[-1, 1:-2]
    b[1:-2, 0] = -gx[1:-2, 0]
    b[1:-2, -1] = gx[1:-2, -1]
    b[0, 0] = (1 / np.sqrt(2)) * (-gy[0, 0] - gx[0, 0])
    b[0, -1] = (1 / np.sqrt(2)) * (-gy[0, -1] + gx[0, -1])
    b[-1, -1] = (1 / np.sqrt(2)) * (gy[-1, -1] + gx[-1, -1])
    b[-1, 0] = (1 / np.sqrt(2)) * (gy[-1, 0] - gx[-1, 0])

    # Modification near the boundaries to enforce the non-homogeneous Neumann BC (Eq. 53 in [1])
    f[0, 1:-2] = f[0, 1:-2] - b[0, 1:-2]
    f[-1, 1:-2] = f[-1, 1:-2] - b[-1, 1:-2]
    f[1:-2, 0] = f[1:-2, 0] - b[1:-2, 0]
    f[1:-2, -1] = f[1:-2, -1] - b[1:-2, -1]

    # Modification near the corners (Eq. 54 in [1])
    f[0, -1] = f[0, -1] - np.sqrt(2) * b[0, -1]
    f[-1, -1] = f[-1, -1] - np.sqrt(2) * b[-1, -1]
    f[-1, 0] = f[-1, 0] - np.sqrt(2) * b[-1, 0]
    f[0, 0] = f[0, 0] - np.sqrt(2) * b[0, 0]

    # Cosine transform of f
    tt = fftpack.dct(f, norm="ortho")
    fcos = fftpack.dct(tt.T, norm="ortho").T

    # Cosine transform of z (Eq. 55 in [1])
    (x, y) = np.meshgrid(range(1, f.shape[1] + 1), range(1, f.shape[0] + 1), copy=True)
    denom = 4 * (
        (np.sin(0.5 * math.pi * x / (f.shape[1]))) ** 2
        + (np.sin(0.5 * math.pi * y / (f.shape[0]))) ** 2
    ).astype(np.float32)

    # Inverse Discrete cosine Transform
    f = -fcos / denom
    tt = fftpack.idct(f, norm="ortho")
    img_tt = fftpack.idct(tt.T, norm="ortho").T
    img_tt = img_tt.mean() + img_tt

    return img_tt
