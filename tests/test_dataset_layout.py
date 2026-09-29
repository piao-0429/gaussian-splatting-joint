"""CPU regressions for DexMirror's aligned COLMAP dataset contract."""
import contextlib
import io
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image as PILImage

from scene import Scene
from scene.dataset_readers import readColmapSceneInfo
from utils.read_write_model import Camera, Image, Point3D, write_model


class DatasetLayoutTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = self.root / 'aligned_sparse/0'
        self.model.mkdir(parents=True)
        for name in ('images', 'images_ft'):
            (self.root / name).mkdir()
        self.cameras = {1: Camera(1, 'PINHOLE', 16, 16, np.array([12., 12., 8., 8.]))}
        self.images = {}
        for index in range(4):
            name = f'view_{index}.png'
            self.images[index + 1] = Image(
                index + 1, np.array([1., 0., 0., 0.]), np.array([index * .1, 0., 0.]),
                1, name, np.empty((0, 2)), np.array([], dtype=np.int64))
            folder = 'images' if index < 2 else 'images_ft'
            PILImage.new('RGB', (16, 16), (100, 120, 140)).save(self.root / folder / name)
        self.points = {
            1: Point3D(1, np.array([0., 0., 2.]), np.array([20, 40, 60]), 0.,
                      np.array([], dtype=np.int32), np.array([], dtype=np.int32)),
            2: Point3D(2, np.array([.1, .2, 2.1]), np.array([80, 100, 120]), 0.,
                      np.array([], dtype=np.int32), np.array([], dtype=np.int32)),
        }
        write_model(self.cameras, self.images, self.points, self.model)

    def read_scene(self, **options):
        settings = dict(path=str(self.root), images='images', depths='', ft_masks='',
                        eval=False, train_test_exp=False)
        settings.update(options)
        with contextlib.redirect_stdout(io.StringIO()):
            return readColmapSceneInfo(**settings)

    def write_legacy_model(self):
        legacy = self.root / 'sparse/0'
        legacy.mkdir(parents=True)
        write_model(self.cameras, self.images, self.points, legacy)
        write_model(self.cameras, self.images, self.points, legacy, ext='.txt')
        return legacy

    def test_aligned_only_loads_views_and_generates_point_cloud(self):
        scene = self.read_scene()
        self.assertEqual(len(scene.train_cameras), 2)
        self.assertEqual(len(scene.finetune_cameras), 2)
        self.assertEqual(scene.num_objects, 0)
        np.testing.assert_allclose(scene.point_cloud.points,
                                   [point.xyz for point in self.points.values()])
        self.assertEqual(Path(scene.ply_path), self.model / 'points3D.ply')
        self.assertTrue(Path(scene.ply_path).is_file())

    def test_sparse_only_dataset_is_rejected_with_preparation_hint(self):
        self.write_legacy_model()
        shutil.rmtree(self.root / 'aligned_sparse')
        with self.assertRaisesRegex(FileNotFoundError, 'aligned_sparse.*cameras.bin.*dataset_preparation'):
            self.read_scene()

    def test_corrupt_aligned_cameras_do_not_fall_back_to_valid_sparse(self):
        self.write_legacy_model()
        (self.model / 'images.bin').write_bytes(b'')
        with self.assertRaisesRegex(ValueError, 'Cannot read aligned COLMAP cameras') as caught:
            self.read_scene()
        self.assertIsNotNone(caught.exception.__cause__)

    def test_corrupt_binary_points_do_not_fall_back_to_aligned_text(self):
        write_model(self.cameras, self.images, self.points, self.model, ext='.txt')
        (self.model / 'points3D.bin').write_bytes(b'')
        with self.assertRaisesRegex(ValueError, 'Cannot read aligned COLMAP points'):
            self.read_scene()

    def test_cached_ply_does_not_hide_missing_binary_reconstruction(self):
        self.read_scene()
        (self.model / 'points3D.bin').unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'Missing: points3D.bin'):
            self.read_scene()

    def test_custom_test_split_uses_the_aligned_reconstruction(self):
        self.write_legacy_model()
        (self.model / 'test.txt').write_text('view_0.png\nview_2.png\n')
        (self.root / 'sparse/0/test.txt').write_text('view_1.png\n')
        scene = self.read_scene(eval=True, llffhold=0)
        self.assertEqual([camera.image_name for camera in scene.test_cameras],
                         ['view_0.png', 'view_2.png'])
        self.assertEqual(len(scene.train_cameras), 1)
        self.assertEqual(len(scene.finetune_cameras), 1)

    def test_blender_only_dataset_is_rejected_by_scene_entrypoint(self):
        shutil.rmtree(self.root / 'aligned_sparse')
        (self.root / 'transforms_train.json').write_text('{"frames": []}')
        args = SimpleNamespace(model_path=str(self.root / 'output'), obj_ply_path='',
                               source_path=str(self.root), images='images', depths='',
                               ft_masks='', eval=False, train_test_exp=False)
        with self.assertRaisesRegex(FileNotFoundError, 'aligned_sparse'):
            Scene(args, SimpleNamespace())


if __name__ == '__main__':
    unittest.main()
