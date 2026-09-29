import math
import json
import random
import sys
import tempfile
import unittest
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from utils.mask_utils import compute_prune_mask, prune_gaussians_with_object_masks
from utils.training_state import CameraSampler, capture_rng, restore_rng, load_checkpoint


class MaskTests(unittest.TestCase):
    def test_projection_votes_match_scalar_reference(self):
        generator = torch.Generator().manual_seed(31)
        points = torch.randn((120, 3), generator=generator)
        points[:4] = torch.tensor([[0., 0., 0.], [0., 0., -1.],
                                   [1., -1., 1.], [-1., 1., 1.]])
        cameras = []
        for offset in (0., .2, -.2):
            projection = torch.eye(4)
            projection[2, 3], projection[3, 3] = 1., 0.
            projection[3, 0] = offset
            mask = torch.rand((1, 9, 11), generator=generator)
            cameras.append(SimpleNamespace(object_masks=[None, mask], image_width=11,
                                           image_height=9, full_proj_transform=projection))
        model = SimpleNamespace(get_xyz=points)
        for expansion in (0., 1.2):
            for proportion in (0., .5, 1.):
                expected = []
                threshold = max(1, math.ceil(proportion * len(cameras)))
                for point in points:
                    votes = 0
                    for camera in cameras:
                        clip = torch.cat((point, torch.ones(1))) @ camera.full_proj_transform
                        if clip[3] <= 0:
                            continue
                        x, y = (clip[:2] / clip[3]).tolist()
                        if not (-1 <= x <= 1 and -1 <= y <= 1):
                            continue
                        x, y = round((x * .5 + .5) * 10), round((y * .5 + .5) * 8)
                        radius = math.ceil(expansion)
                        patch = camera.object_masks[1][0, max(0, y-radius):y+radius+1,
                                                       max(0, x-radius):x+radius+1]
                        votes += bool((patch > .5).any())
                    expected.append(votes < threshold)
                actual, used = compute_prune_mask(model, cameras, proportion, .5, expansion, 1)
                self.assertEqual(used, threshold)
                self.assertTrue(torch.equal(actual, torch.tensor(expected)))

    def test_threshold_uses_all_mask_views_and_cache_invalidates(self):
        point = SimpleNamespace(get_xyz=torch.zeros((1, 3)))
        camera = SimpleNamespace(object_masks=[torch.ones((1, 3, 3))], image_width=3,
                                 image_height=3, full_proj_transform=torch.eye(4))
        hidden = SimpleNamespace(**vars(camera))
        hidden.full_proj_transform = -torch.eye(4)
        missing = SimpleNamespace(object_masks=[None])
        mask, threshold = compute_prune_mask(point, [camera, hidden, missing], 1.)
        self.assertEqual(threshold, 2)
        self.assertTrue(mask.item())
        cache = {}
        self.assertFalse(compute_prune_mask(point, [camera], cache=cache)[0].item())
        camera.object_masks[0].zero_()
        self.assertTrue(compute_prune_mask(point, [camera], cache=cache)[0].item())
        for model in (point, SimpleNamespace(get_xyz=torch.empty((0, 3)))):
            self.assertEqual(prune_gaussians_with_object_masks(model, []), (0, 0))


class StateTests(unittest.TestCase):
    def test_inference_config_preserves_pipeline_and_cli_overrides(self):
        from arguments import ModelParams, PipelineParams, get_combined_args
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            config = {'model': {'source_path': '/original', 'resolution': 2},
                      'pipeline': {'antialiasing': True, 'compute_cov3D_python': True}}
            (path / 'training_config.json').write_text(json.dumps(config))
            parser = ArgumentParser()
            ModelParams(parser, sentinel=True)
            PipelineParams(parser)
            with patch.object(sys, 'argv', ['render.py', '-m', directory, '-s', '/override']):
                args = get_combined_args(parser)
            self.assertTrue(args.antialiasing)
            self.assertTrue(args.compute_cov3D_python)
            self.assertEqual(args.resolution, 2)
            self.assertEqual(args.source_path, '/override')
            (path / 'training_config.json').unlink()
            (path / 'cfg_args').write_text("Namespace(source_path='/legacy', resolution=1)")
            with patch.object(sys, 'argv', ['render.py', '-m', directory]):
                self.assertEqual(get_combined_args(parser).source_path, '/legacy')

    def test_sampling_matches_baseline_and_resumes_across_refills(self):
        cameras = [SimpleNamespace(image_name=str(i)) for i in range(7)]
        random.seed(41)
        stack, expected = [], []
        for _ in range(20):
            if not stack:
                stack = cameras.copy()
            expected.append(stack.pop(random.randint(0, len(stack)-1)).image_name)
        random.seed(41)
        sampler = CameraSampler(cameras)
        self.assertEqual([sampler.sample().image_name for _ in range(20)], expected)
        state, rng = sampler.state_dict(), capture_rng()
        expected = [sampler.sample().image_name for _ in range(20)]
        resumed = CameraSampler(list(reversed(cameras)))
        resumed.load_state_dict(state)
        restore_rng(rng)
        self.assertEqual([resumed.sample().image_name for _ in range(20)], expected)
        with self.assertRaisesRegex(ValueError, "camera pool"):
            CameraSampler(cameras[:-1]).load_state_dict(state)
        self.assertIsNone(CameraSampler([]).sample())

    def test_legacy_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'old.pth'
            torch.save(((), 12), path)
            with self.assertRaisesRegex(ValueError, 'background-only'):
                load_checkpoint(path)


if __name__ == '__main__':
    unittest.main()
