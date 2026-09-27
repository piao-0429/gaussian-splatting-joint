"""Joint Gaussian rendering entry point; implementation shared across modes."""

from utils.joint_utils import (
    DEFAULT_MOVE_R, DEFAULT_MOVE_T, cam_has_mask, load_object_gaussians,
    merge_gaussians, move, render_view,
)
from utils.joint_render import main, save_image


if __name__ == "__main__":
    main(objects_only=False)
