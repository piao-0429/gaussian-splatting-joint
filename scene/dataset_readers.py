#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from typing import NamedTuple
from scene.colmap_loader import (qvec2rotmat, read_extrinsics_binary,
                                 read_intrinsics_binary, read_points3D_binary)
from utils.graphics_utils import getWorld2View2, focal2fov
import numpy as np
import json
from plyfile import PlyData, PlyElement
from scene.gaussian_model import BasicPointCloud

MASK_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    depth_params: dict
    image_path: str
    image_name: str
    depth_path: str
    mask_paths: list  # One entry per object; empty string if missing
    width: int
    height: int
    is_test: bool
    is_finetune: bool = False

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    finetune_cameras: list
    nerf_normalization: dict
    ply_path: str
    num_objects: int

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, depths_params, images_folder, depths_folder, object_mask_folders, test_cam_names_list):
    cam_infos = []

    def find_mask(base_dir: str, image_name: str, stem_name: str):
        if not base_dir:
            return ""
        candidates = [
            os.path.join(base_dir, image_name),
            os.path.join(base_dir, os.path.basename(image_name)),
        ]
        for ext in MASK_EXTENSIONS:
            candidates.append(os.path.join(base_dir, f"{stem_name}{ext}"))
            candidates.append(os.path.join(base_dir, f"{os.path.basename(stem_name)}{ext}"))
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return ""

    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        n_remove = len(extr.name.split('.')[-1]) + 1
        depth_params = None
        if depths_params is not None:
            try:
                depth_params = depths_params[extr.name[:-n_remove]]
            except:
                print("\n", key, "not found in depths_params")

        image_path = os.path.join(images_folder, extr.name)
        if not os.path.exists(image_path):
            image_path = os.path.join(images_folder.replace("images", "images_ft"), os.path.basename(extr.name))
        # if not os.path.exists(image_path):
        #     image_path = os.path.join(images_folder.replace("images", "images_ft_obj"), os.path.basename(extr.name))
        image_name = extr.name
        depth_path = os.path.join(depths_folder, f"{extr.name[:-n_remove]}.png") if depths_folder != "" else ""
        stem_name = extr.name[:-n_remove]

        mask_paths = []
        is_finetune = "images_ft" in image_path
        if is_finetune:
            for obj_dir in object_mask_folders:
                mask_paths.append(find_mask(obj_dir, extr.name, stem_name))
        else:
            mask_paths = [""] * len(object_mask_folders)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, depth_params=depth_params,
                              image_path=image_path, image_name=image_name, depth_path=depth_path,
                              mask_paths=mask_paths,
                              width=width, height=height, is_test=image_name in test_cam_names_list,
                              is_finetune=is_finetune)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, depths, ft_masks, eval, train_test_exp, llffhold=8):
    model_dir = os.path.join(path, "aligned_sparse", "0")
    required = ("cameras.bin", "images.bin", "points3D.bin")
    missing = [name for name in required if not os.path.isfile(os.path.join(model_dir, name))]
    if missing:
        raise FileNotFoundError(
            f"DexMirror requires an aligned COLMAP reconstruction in {model_dir!r}. "
            f"Missing: {', '.join(missing)}. Prepare cameras.bin, images.bin and "
            "points3D.bin from the same reconstruction; see docs/dataset_preparation.md.")
    try:
        cam_extrinsics = read_extrinsics_binary(os.path.join(model_dir, "images.bin"))
        cam_intrinsics = read_intrinsics_binary(os.path.join(model_dir, "cameras.bin"))
    except Exception as error:
        raise ValueError(f"Cannot read aligned COLMAP cameras from {model_dir!r}") from error

    depth_params_file = os.path.join(model_dir, "depth_params.json")
    ## if depth_params_file isnt there AND depths file is here -> throw error
    depths_params = None
    if depths != "":
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale

        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)

    if eval:
        if llffhold:
            print("------------LLFF HOLD-------------")
            cam_names = [cam_extrinsics[cam_id].name for cam_id in cam_extrinsics]
            cam_names = sorted(cam_names)
            test_cam_names_list = [name for idx, name in enumerate(cam_names) if idx % llffhold == 0]
        else:
            with open(os.path.join(model_dir, "test.txt"), 'r') as file:
                test_cam_names_list = [line.strip() for line in file]
    else:
        test_cam_names_list = []

    reading_dir = "images" if images == None else images

    ft_masks_dir = ""
    if ft_masks:
        if os.path.isabs(ft_masks):
            ft_masks_dir = ft_masks if os.path.isdir(ft_masks) else ""
        else:
            candidate = os.path.join(path, ft_masks)
            ft_masks_dir = candidate if os.path.isdir(candidate) else ""

    object_mask_folders = []
    if ft_masks_dir:
        subdirs = [os.path.join(ft_masks_dir, d) for d in sorted(os.listdir(ft_masks_dir)) if os.path.isdir(os.path.join(ft_masks_dir, d))]
        if subdirs:
            object_mask_folders = subdirs
        else:
            object_mask_folders = [ft_masks_dir]

    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, depths_params=depths_params,
        images_folder=os.path.join(path, reading_dir), 
        depths_folder=os.path.join(path, depths) if depths != "" else "",
        object_mask_folders=object_mask_folders,
        test_cam_names_list=test_cam_names_list)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    # Report how many masks were found per object-folder
    if object_mask_folders:
        counts = [0] * len(object_mask_folders)
        for c in cam_infos:
            for i, p in enumerate(c.mask_paths):
                if p:
                    counts[i] += 1

        print("[INFO] Finetune mask summary:")
        for i, folder in enumerate(object_mask_folders):
            name = os.path.basename(folder.rstrip(os.sep)) or folder
            print(f"  Object {i}: folder='{folder}' (name='{name}') -> {counts[i]} masks matched")

    train_cam_infos = []
    finetune_cam_infos = []
    test_cam_infos = []
    
    for c in cam_infos:
        if train_test_exp or not c.is_test:
            if c.is_finetune:
                finetune_cam_infos.append(c)
            else:
                train_cam_infos.append(c)
        if c.is_test:
            test_cam_infos.append(c)

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(model_dir, "points3D.ply")
    if not os.path.exists(ply_path):
        bin_path = os.path.join(model_dir, "points3D.bin")
        print("Converting aligned points3D.bin to points3D.ply.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except Exception as error:
            raise ValueError(f"Cannot read aligned COLMAP points from {bin_path!r}") from error
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except Exception as error:
        raise ValueError(f"Cannot read aligned point cloud {ply_path!r}") from error

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           finetune_cameras=finetune_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           num_objects=len(object_mask_folders))
    return scene_info
