"""Regression coverage for evaluation eval_412bc052c68847's byte-image blocker."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy.smolvla.deploy_policy import encode_obs
from policy.smolvla.smolvla_worker import prepare_observation


def camera_observation():
    """Simulator-shaped, nonconstant RGBA bytes, including pixel values 0 and 255."""
    rgb = (np.arange(480 * 640 * 3) % 256).astype(np.uint8).reshape(480, 640, 3)
    rgba = np.concatenate((rgb, np.full((480, 640, 1), 255, np.uint8)), axis=-1)
    return {
        "sensor": {name: {"color": rgba[None].copy()} for name in
                   ("cam_high", "cam_left_wrist", "cam_right_wrist")},
        "robot": {name: np.linspace(-2, 2, 14, dtype=np.float32)[None] for name in
                  ("qpos", "qvel", "qf")},
    }


class ImageContractTest(unittest.TestCase):
    def test_upstream_image_contract_before_and_after_fix(self):
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        encoded = encode_obs(camera_observation())
        keys = [key for key in encoded if key.startswith("observation.images.")]
        policy = SimpleNamespace(config=SimpleNamespace(
            image_features=keys, resize_imgs_with_padding=[512, 512], empty_cameras=0))
        original = {key: torch.from_numpy(encoded[key])[None] for key in keys}
        # CPU bilinear kernels can accept bytes even though the organizer's
        # CUDA kernel rejects them. On CPU the original path still violates
        # the contract: SigLIP receives values outside its [-1, 1] range.
        original_images, _ = SmolVLAPolicy.prepare_images(policy, original)
        self.assertTrue(any((image > 1).any() for image in original_images))
        prepared = prepare_observation(encoded, "task")
        images, masks = SmolVLAPolicy.prepare_images(
            policy, {key: prepared[key][None] for key in keys})
        self.assertEqual(len(images), 3)
        for image, mask in zip(images, masks):
            self.assertEqual(image.dtype, torch.float32)
            self.assertTrue(torch.isfinite(image).all())
            self.assertTrue(((image >= -1) & (image <= 1)).all())
            self.assertTrue(mask.all())

    def test_camera_bytes_resize_and_state_units(self):
        raw = camera_observation()
        encoded = encode_obs(raw)
        prepared = prepare_observation(encoded, "open drawer")
        for key in (k for k in encoded if k.startswith("observation.images.")):
            self.assertEqual(encoded[key].dtype, np.uint8)
            self.assertEqual(prepared[key].dtype, torch.float32)
            self.assertEqual(tuple(prepared[key].shape), (3, 480, 640))
            expected = torch.from_numpy(raw["sensor"]["cam_high"]["color"][0, :, :, :3])
            torch.testing.assert_close(prepared[key], expected.permute(2, 0, 1).float() / 255)
            resized = F.interpolate(prepared[key][None], size=(384, 512), mode="bilinear")
            self.assertTrue(torch.isfinite(resized).all())
        for key in ("observation.state", "observation.qvel", "observation.qf"):
            torch.testing.assert_close(prepared[key], torch.from_numpy(encoded[key]))
        self.assertEqual(prepared["task"], "open drawer")

    def test_normalized_float_images_are_not_scaled_twice(self):
        image = np.array([0, 0.25, 0.5, 1], dtype=np.float32).reshape(1, 2, 2)
        prepared = prepare_observation({"observation.images.cam_high": image}, "task")
        torch.testing.assert_close(prepared["observation.images.cam_high"], torch.from_numpy(image))

    def test_invalid_image_ranges_fail_explicitly(self):
        for values in ([0, 255], [-1, 1], [0, float("nan")], [0, float("inf")]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                prepare_observation({"observation.images.cam_high": np.array(values, np.float32)}, "task")
        with self.assertRaises(TypeError):
            prepare_observation({"observation.images.cam_high": np.array([1], np.int16)}, "task")


if __name__ == "__main__":
    unittest.main()
