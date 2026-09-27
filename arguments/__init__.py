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

from argparse import ArgumentParser, Namespace
import ast
import json
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._obj_ply_path = ""
        self._images = "images"
        self._depths = ""
        self.ft_masks = "masks_ft_obj"
        self._resolution = -1
        self._white_background = False
        self.train_test_exp = False
        self.data_device = "cuda"
        self.camera_workers = 4
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        g.obj_ply_path = os.path.abspath(g.obj_ply_path) if g.obj_ply_path else ""
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.object_only_until_iter = 0
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.025
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.exposure_lr_init = 0.01
        self.exposure_lr_final = 0.001
        self.exposure_lr_delay_steps = 0
        self.exposure_lr_delay_mult = 0.0
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.depth_l1_weight_init = 1.0
        self.depth_l1_weight_final = 0.01
        self.random_background = False
        self.optimizer_type = "default"
        self.log_interval = 10
        # Proportion (0.0-1.0) of available mask views required to keep a Gaussian
        self.mask_prune_min_prop = 0.5
        self.mask_prune_threshold = 0.5
        # Dilation radius (in pixels) applied to masks before pruning; 0 disables
        self.mask_prune_expand = 0.0
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    args = parser.parse_args(sys.argv[1:])
    root = args.model_path or ""
    config_path = os.path.join(root, "training_config.json")
    if os.path.isfile(config_path):
        with open(config_path) as file:
            config = json.load(file)
        defaults = dict(config["model"], **config["pipeline"])
    else:
        config_path = os.path.join(root, "cfg_args")
        with open(config_path) as file:
            expression = ast.parse(file.read().strip(), mode="eval").body
        if not (isinstance(expression, ast.Call) and isinstance(expression.func, ast.Name)
                and expression.func.id == "Namespace" and not expression.args):
            raise ValueError("Invalid cfg_args: expected Namespace keyword literals")
        defaults = {arg.arg: ast.literal_eval(arg.value) for arg in expression.keywords}
    print("Config file found:", config_path)
    known = {action.dest for action in parser._actions}
    parser.set_defaults(**{key: value for key, value in defaults.items() if key in known})
    return parser.parse_args(sys.argv[1:])
