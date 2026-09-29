import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image


class CameraLoadingTests(unittest.TestCase):
    def test_parallel_camera_loading_preserves_input_order(self):
        import time
        from utils.camera_utils import cameraList_from_camInfos
        cameras = [SimpleNamespace(image_name=str(i), image_path=str(i)) for i in range(8)]
        def load(args, index, camera, *unused):
            time.sleep((8-index) * .001)
            return camera
        with patch('utils.camera_utils.loadCam', side_effect=load):
            result = cameraList_from_camInfos(cameras, 1., SimpleNamespace(camera_workers=4), False, False)
        self.assertEqual(result, cameras)

    def test_camera_failure_is_reported_instead_of_silently_dropped(self):
        from utils.camera_utils import cameraList_from_camInfos
        camera = SimpleNamespace(image_name='broken.png', image_path='/data/broken.png')
        with patch('utils.camera_utils.loadCam', side_effect=OSError('corrupt image')):
            with self.assertRaisesRegex(RuntimeError, 'broken.png.*corrupt image'):
                cameraList_from_camInfos([camera], 1., None, False, False)


@unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA')
class RuntimeGPUTests(unittest.TestCase):
    def test_rgb_camera_omits_identity_alpha_and_keeps_binary_masks(self):
        from scene.cameras import Camera
        image = Image.fromarray(np.full((8, 10, 3), 127, dtype=np.uint8))
        mask = Image.fromarray(np.eye(8, 10, dtype=np.uint8) * 255)
        kwargs = dict(resolution=(10, 8), colmap_id=0, R=np.eye(3), T=np.zeros(3),
                      FoVx=1., FoVy=1., depth_params=None, image=image, image_name='test',
                      mask_images=[mask], invdepthmap=np.ones((8, 10), dtype=np.float32))
        camera = Camera(**kwargs)
        self.assertIsNone(camera.alpha_mask)
        self.assertEqual(camera.object_masks[0].dtype, torch.bool)
        self.assertEqual(camera.depth_mask.dtype, torch.bool)
        self.assertTrue(torch.equal(camera.object_masks[0].cpu(), torch.eye(8, 10).bool()[None]))
        exposure = Camera(**kwargs, train_test_exp=True, is_test_view=True, is_test_dataset=False)
        self.assertTrue(torch.all(exposure.alpha_mask[..., :5] == 1))
        self.assertTrue(torch.all(exposure.alpha_mask[..., 5:] == 0))

    def test_dense_statistics_equal_original_indexed_updates(self):
        from scene import GaussianModel
        model = GaussianModel(0)
        model.xyz_gradient_accum = torch.rand((17, 1), device='cuda')
        model.denom = torch.zeros((17, 1), device='cuda')
        points = torch.zeros((17, 3), device='cuda', requires_grad=True)
        points.grad = torch.randn_like(points)
        visible = torch.arange(17, device='cuda') % 3 != 0
        expected = model.xyz_gradient_accum.clone()
        indices = visible.nonzero()
        expected[indices] += points.grad[indices, :2].norm(dim=-1, keepdim=True)
        model.add_densification_stats(points, visible)
        torch.testing.assert_close(model.xyz_gradient_accum, expected, rtol=0, atol=0)
        torch.testing.assert_close(model.denom, visible[:, None].float(), rtol=0, atol=0)

    def test_paired_evaluation_uses_one_render_and_preserves_metrics(self):
        import eval as evaluation
        from utils.image_utils import psnr
        torch.manual_seed(4)
        pred, gt = torch.rand((3, 16, 16), device='cuda'), torch.rand((3, 16, 16), device='cuda')
        mask = torch.rand((1, 16, 16), device='cuda') > .5
        camera = SimpleNamespace(image_name='view', object_masks=[mask])
        modes = [{'name': 'masked', 'mask_index': 0, 'mask_prediction': True},
                 {'name': 'unmasked', 'mask_index': 0, 'mask_prediction': False}]
        with patch.object(evaluation, 'render_view', return_value=(pred, gt)) as renderer:
            results = evaluation._evaluate_modes([camera], None, None, None, False, False,
                                                 modes, sample_count=0)
        renderer.assert_called_once()
        for result, prediction in zip(results, (pred * mask, pred)):
            self.assertAlmostEqual(result['l1'], (prediction-gt*mask).abs().mean().item(), places=7)
            self.assertAlmostEqual(result['psnr'], psnr(prediction, gt*mask).mean().item(), places=6)

    def test_render_retention_changes_no_pixels_or_parameter_gradients(self):
        from scene import GaussianModel
        from scene.cameras import Camera
        from utils.graphics_utils import BasicPointCloud
        from gaussian_renderer import render
        rng = np.random.default_rng(2)
        xyz = rng.uniform([-.5, -.5, 1.8], [.5, .5, 2.2], (32, 3))
        model = GaussianModel(0)
        model.create_from_pcd(BasicPointCloud(xyz, rng.random((32, 3)), np.zeros((32, 3))), [], 1.)
        camera = Camera((32, 32), 0, np.eye(3), np.zeros(3), 1., 1., None,
                        Image.new('RGB', (32, 32)), image_name='view')
        pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False,
                               antialiasing=False, debug=False)
        background = torch.zeros(3, device='cuda')
        a = render(camera, model, pipe, background, retain_viewspace_grad=True)
        b = render(camera, model, pipe, background, retain_viewspace_grad=False)
        self.assertEqual(a['visibility_filter'].dtype, torch.bool)
        self.assertFalse(b['viewspace_points'].requires_grad)
        torch.testing.assert_close(a['render'], b['render'], rtol=0, atol=0)
        params = [model._xyz, model._features_dc, model._opacity, model._scaling, model._rotation]
        ga = torch.autograd.grad(a['render'].sum(), params)
        gb = torch.autograd.grad(b['render'].sum(), params)
        for x, y in zip(ga, gb):
            torch.testing.assert_close(x, y, rtol=1e-5, atol=1e-5)
        with torch.no_grad():
            self.assertFalse(render(camera, model, pipe, background)['viewspace_points'].requires_grad)


if __name__ == '__main__':
    unittest.main()
