"""Small real-rasterizer regressions; run in the project's CUDA environment."""
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from argparse import ArgumentParser
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch
from PIL import Image as PILImage
from plyfile import PlyData

REPO = Path(__file__).resolve().parents[1]


def make_dataset(root, objects=2):
    from scene.dataset_readers import storePly
    from utils.read_write_model import Camera, Image, Point3D, write_model
    for folder in ['aligned_sparse/0', 'images', 'images_ft', 'depth']:
        (root / folder).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    xyz = rng.uniform([-.45, -.25, 1.8], [.45, .25, 2.2], (64, 3))
    rgb = rng.integers(40, 220, (64, 3), dtype=np.uint8)
    points = {i+1: Point3D(i+1, p, c, 0., np.array([], dtype=np.int32), np.array([], dtype=np.int32))
              for i, (p, c) in enumerate(zip(xyz, rgb))}
    cameras = {1: Camera(1, 'PINHOLE', 64, 64, np.array([48., 48., 32., 32.]))}
    images, depths = {}, {}
    yy, xx = np.mgrid[:64, :64]
    for i in range(16 if objects else 8):
        filename = 'view_{:02d}.png'.format(i)
        images[i+1] = Image(i+1, np.array([1., 0., 0., 0.]),
                            np.array([(-1)**i * .2, 0., 0.]), 1, filename,
                            np.empty((0, 2)), np.array([], dtype=np.int64))
        pixels = np.stack([xx * 3, yy * 3, np.full_like(xx, 110)], axis=-1).astype(np.uint8)
        PILImage.fromarray(pixels).save(root / ('images_ft' if i >= 8 else 'images') / filename)
        PILImage.fromarray(np.full((64, 64), 32768, dtype=np.uint16)).save(root / 'depth' / filename)
        depths[filename[:-4]] = {'scale': 1., 'offset': 0.}
        if i >= 8:
            for index in range(objects):
                folder = root / 'masks_ft_obj' / str(index)
                folder.mkdir(parents=True, exist_ok=True)
                mask = xx < 32 if index == 0 else xx >= 32
                PILImage.fromarray(mask.astype(np.uint8) * 255).save(folder / filename)
    write_model(cameras, images, points, root / 'aligned_sparse/0')
    storePly(str(root / 'aligned_sparse/0/points3D.ply'), xyz, rgb)
    (root / 'aligned_sparse/0/depth_params.json').write_text(json.dumps(depths))
    return root


@unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA and the Gaussian rasterizer')
class JointGPUTests(unittest.TestCase):
    @classmethod
    def run_cli(cls, name, script, args):
        result = subprocess.run([sys.executable, str(REPO / script)] + list(map(str, args)),
                                cwd=REPO, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=120)
        (cls.root / (name + '.log')).write_text(result.stdout)
        if result.returncode:
            raise AssertionError(result.stdout[-8000:])
        return result.stdout

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='joint-regression-')
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.dataset = make_dataset(cls.root / 'data')
        cls.output = cls.root / 'continuous'
        cls.common = ['-s', cls.dataset, '-d', 'depth', '--eval', '--iterations', 12,
                      '--object_only_until_iter', 4, '--densify_from_iter', 1,
                      '--densify_until_iter', 8, '--densification_interval', 2,
                      '--opacity_reset_interval', 1000, '--prune_iterations', 6, 12,
                      '--checkpoint_iterations', 2, 6, 12, '--test_iterations', 12,
                      '--save_iterations', 12, '--random_background', '--disable_viewer',
                      # Spherical initialization has near-zero rotation gradients;
                      # Adam amplifies atomic roundoff into different split points.
                      # Freeze rotations for the trajectory comparison only.
                      '--rotation_lr', 0]
        cls.log = cls.run_cli('continuous', 'train.py', ['-m', cls.output] + cls.common)

    def assert_state_equal(self, a, b, exact=False):
        if isinstance(a, torch.Tensor):
            # CUDA rasterization accumulates gradients with floating-point
            # atomics. Counters/RNG must match exactly; floats may differ by ULPs.
            tolerance = dict(rtol=1e-5, atol=2e-7) if a.is_floating_point() and not exact else dict(rtol=0, atol=0)
            torch.testing.assert_close(a, b, **tolerance)
        elif isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                self.assert_state_equal(a[key], b[key], exact)
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assert_state_equal(x, y, exact)
        else:
            self.assertEqual(a, b)

    def test_resume_before_and_after_phase_transition(self):
        continuous = torch.load(self.output / 'chkpnt12.pth')
        self.assertIn('Evaluating test_scene_plus_objects:', self.log)
        self.assertIn('Mask pruning removed', self.log)
        self.assertTrue(continuous['objects_finetuned'])
        self.assertTrue(any(len(obj['gaussians'][1]) != 64 for obj in continuous['objects']))
        for iteration in (2, 6):
            output = self.root / ('resumed_' + str(iteration))
            self.run_cli('resume_' + str(iteration), 'train.py',
                         ['--start_checkpoint', self.output / ('chkpnt%d.pth' % iteration), '-m', output])
            resumed = torch.load(output / 'chkpnt12.pth')
            for key in ('background', 'objects', 'samplers', 'rng', 'ema',
                        'objects_finetuned', 'scene_optim_initialized'):
                self.assert_state_equal(continuous[key], resumed[key])
            for source in (self.output / 'point_cloud/iteration_12').glob('*.ply'):
                a = PlyData.read(str(source))['vertex'].data
                b = PlyData.read(str(output / 'point_cloud/iteration_12' / source.name))['vertex'].data
                self.assertEqual(a.shape, b.shape)
                for field in a.dtype.names:
                    np.testing.assert_allclose(a[field], b[field], rtol=1e-5, atol=2e-7)

    def test_model_restore_preserves_every_checkpoint_tensor_exactly(self):
        from argparse import Namespace
        from scene import GaussianModel
        for iteration in (2, 6):
            state = torch.load(self.output / ('chkpnt%d.pth' % iteration))
            opt = Namespace(**state['config']['optimization'])
            bg = GaussianModel(3)
            bg.restore_full(state['background'], opt)
            self.assert_state_equal(state['background'], bg.capture_full(), exact=True)
            for saved in state['objects']:
                obj = GaussianModel(3)
                obj.restore_full(saved, opt, 'finetune' if state['objects_finetuned'] else 'object')
                self.assert_state_equal(saved, obj.capture_full(include_exposure=False), exact=True)

    def test_exposure_resume_and_evaluation(self):
        source, resumed = self.root / 'exposure', self.root / 'exposure_resume'
        self.run_cli('exposure', 'train.py', ['-m', source] + self.common +
                     ['--train_test_exp', '--iterations', 5, '--checkpoint_iterations', 2, 5,
                      '--prune_iterations', 5, '--test_iterations', 5])
        self.run_cli('exposure_resume', 'train.py',
                     ['--start_checkpoint', source / 'chkpnt2.pth', '-m', resumed])
        a, b = torch.load(source / 'chkpnt5.pth'), torch.load(resumed / 'chkpnt5.pth')
        self.assert_state_equal(a['background'], b['background'])
        self.assert_state_equal(a['objects'], b['objects'])
        self.assertTrue(a['background']['exposure']['optimizer']['state'])
        self.assertEqual(len(a['background']['exposure']['mapping']), 16)
        self.run_cli('exposure_eval', 'eval.py', ['-m', source])

    def test_gaussian_ply_initialization_and_background_only(self):
        from arguments import OptimizationParams
        from scene import GaussianModel
        path = self.output / 'point_cloud/iteration_12/obj.ply'
        obj = GaussianModel(3)
        obj.load_obj_ply(str(path))
        obj.obj_training_setup(OptimizationParams(ArgumentParser()))
        self.assertGreater(obj.optimizer.param_groups[0]['lr'], 0.)
        self.assertEqual(len(obj.max_radii2D), len(obj.get_xyz))
        self.assertEqual(obj.max_radii2D.device, obj.get_xyz.device)
        self.run_cli('ply_init', 'train.py', ['-m', self.root / 'ply_init'] + self.common +
                     ['-o', path, '--iterations', 5, '--test_iterations', 5])
        data = make_dataset(self.root / 'background_data', objects=0)
        output = self.root / 'background'
        self.run_cli('background', 'train.py', ['-m', output, '-s', data, '--iterations', 3,
                     '--disable_viewer', '--checkpoint_iterations', 3, '--test_iterations', 3])
        self.assertEqual(torch.load(output / 'chkpnt3.pth')['objects'], [])
        self.assertEqual([p.name for p in (output / 'point_cloud/iteration_3').glob('*.ply')],
                         ['point_cloud.ply'])

    def test_holdout_evaluation_and_legacy_render(self):
        self.run_cli('eval', 'eval.py', ['-m', self.output])
        metrics = json.loads((self.output / 'eval/metrics_iter12.json').read_text())
        counts = {entry['name']: entry['count'] for entry in metrics['results']}
        self.assertEqual(counts['test_scene'], 1)
        self.assertEqual(counts['test_scene_plus_objects'], 1)
        self.assertEqual(counts['finetune_obj0_masked'], 7)
        self.assertEqual(counts['finetune_obj1_masked'], 7)
        self.run_cli('legacy_render', 'render.py', ['-m', self.output, '--skip_train'])
        self.assertEqual(len(list((self.output / 'test/ours_12/renders').glob('*.png'))), 2)

    def test_saved_metadata_restores_sh_degree_and_iteration_exposure(self):
        from argparse import Namespace
        from scene import Scene, GaussianModel
        folder = self.root / 'metadata'
        saved = self.output / 'point_cloud/iteration_12'
        shutil.copytree(saved, folder / 'point_cloud/iteration_12')
        (folder / 'exposure.json').write_text('{"wrong_iteration": [[0, 0, 0, 0]]}')
        model = json.loads((self.output / 'training_config.json').read_text())['model']
        model.update(model_path=str(folder), train_test_exp=True)
        scene = Scene(Namespace(**model), GaussianModel(3), load_iteration=12, shuffle=False)
        self.assertEqual(scene.gaussians.active_sh_degree, 0)
        self.assertEqual(scene.num_objects, 2)
        exposures = json.loads((saved / 'exposure.json').read_text())
        self.assertEqual(set(scene.gaussians.pretrained_exposures), set(exposures))

    def test_object_only_renders_once_per_view_without_grad(self):
        import utils.joint_render as cli
        import utils.joint_utils as shared
        flags = []
        original = shared.render

        def observed(*args, **kwargs):
            flags.append(torch.is_grad_enabled())
            return original(*args, **kwargs)

        root = self.root / 'only_objects'
        argv = ['render_joint_only_obj.py', '-m', str(self.output), '--split', 'test',
                '--no_move', '--output_root', str(root)]
        with patch.object(sys, 'argv', argv), patch.object(cli, 'safe_state'), \
                patch.object(shared, 'render', side_effect=observed):
            cli.main(objects_only=True)
        self.assertEqual(flags, [False, False])
        self.assertEqual(len(list(root.rglob('*.png'))), 2)
        self.assertFalse((root / 'test_merged').exists())

    def test_logging_interval_and_single_save_per_model(self):
        import train
        from arguments import ModelParams, OptimizationParams, PipelineParams
        from scene import GaussianModel
        parser = ArgumentParser()
        lp, op, pp = ModelParams(parser), OptimizationParams(parser), PipelineParams(parser)
        args = parser.parse_args(['-s', str(self.dataset), '-m', str(self.root / 'logged'),
                                  '-d', 'depth', '--eval', '--iterations', '5',
                                  '--object_only_until_iter', '4', '--log_interval', '3'])
        writer, saved = Mock(), []
        original = GaussianModel.save_ply

        def save(model, path):
            saved.append(Path(path).name)
            return original(model, path)

        with patch.object(train, 'TENSORBOARD_FOUND', True), \
                patch.object(train, 'SummaryWriter', return_value=writer, create=True), \
                patch.object(GaussianModel, 'save_ply', save):
            train.training(lp.extract(args), op.extract(args), pp.extract(args),
                           [2, 5], [5], [5], [], None, -1, disable_viewer=True)
        scalar_steps = [(call.args[0], call.args[2]) for call in writer.add_scalar.call_args_list]
        self.assertEqual([step for name, step in scalar_steps if name == 'iter_time'], [3, 5])
        self.assertIn(('test_scene_plus_objects/loss_viewpoint - l1_loss', 2), scalar_steps)
        self.assertEqual(sorted(saved), ['obj.ply', 'obj_1.ply', 'point_cloud.ply'])
        writer.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
