# Reconstruction evaluation

[← README](README.md#evaluation)

Use [eval.py](eval.py) to inspect a saved DexMirror reconstruction. It evaluates the background, the background composed with all objects, and individual objects in their reconstructed positions.

## Run evaluation

Activate the reconstruction environment, open the repository, and select a model directory. Keep the source dataset, masks, and saved model configuration available.

```bash
conda activate dexmirror
cd /absolute/path/to/gaussian-splatting-joint
export MODEL_DIR=/absolute/path/to/output/my_scene

python eval.py -m "$MODEL_DIR" --iteration 100000 --sample_count 6
```

Use `--iteration -1` (the default) to load the latest saved iteration. The script reads the training configuration from the model directory. If you moved the dataset, update its paths with `-s`, `--ft_masks`, and `-d` as applicable.

`--sample_count` controls only the number of preview images saved per group. All eligible views contribute to the metrics. Set it to `0` to write metrics without generating new preview images.

## Evaluate held-out views

Add `--eval` to the [training command](README.md#run-dexmirror) **before training**. With the default split settings, the loader sorts registered camera names and reserves every eighth view for testing. Background and object-containing test views are evaluated separately.

For a model trained this way, evaluate only the test groups with:

```bash
python eval.py -m "$MODEL_DIR" --iteration 100000 \
  --skip_train --skip_finetune --sample_count 6
```

Without `--eval` during training, the scores describe training-view reconstruction; adding it only when evaluating does not make those views unseen. A dataset may have no test views in one of its groups, in which case that group has no scores.

<details>
<summary>Advanced: train/test exposure mode</summary>

The description above assumes the default `train_test_exp=False`. With `--train_test_exp`, test cameras also participate in training and evaluation uses the right half of each image. Report this setting when comparing runs.

</details>

## What each group measures

| Group in the output | Model and views |
| --- | --- |
| `train_scene` | Background model on background training views. |
| `test_scene` | Background model on background test views. |
| `test_scene_plus_objects` | Background and all objects on object-containing test views. |
| `finetune_scene_plus_objects` | Background and all objects on object-containing training views. |
| `finetune_obj<N>_masked` | Object `N` alone, with both render and ground truth masked. |
| `finetune_obj<N>_unmasked` | Object `N` alone, with only ground truth masked; the score also reflects content rendered outside its mask. |

The individual-object groups use object-containing **training views** with an available mask for that object. Object indices start at zero and follow the sorted mask-directory names.

To reduce evaluation work, use `--skip_train` for background training views, `--skip_test` for test views, or `--skip_objects` for individual-object groups. `--skip_finetune` skips both the composed training-view group and all individual-object groups.

## Read the outputs

Results are written under `$MODEL_DIR/eval/`. For example:

```text
eval/
├── metrics_iter100000.json
├── train_scene/
│   ├── per_view_metrics.json
│   └── <image_name>.png
├── test_scene/
├── test_scene_plus_objects/
├── finetune_scene_plus_objects/
├── finetune_obj0_masked/
└── finetune_obj0_unmasked/
```

Only evaluated groups with usable views receive per-view files. The aggregate JSON records the iteration, timestamp, and each group's view `count` and mean `l1`, `psnr`, and `ssim`. Groups with no usable views have `count: 0` and no metric values.

Each preview shows **ground truth | render | absolute error**, from left to right. Use it to inspect where errors occur, then consult `per_view_metrics.json` for that image's scores.

The aggregate filename includes the iteration. Per-view JSON files and preview directories are shared between evaluations and may be overwritten or retain older previews. Archive the entire `eval/` directory before evaluating another checkpoint if you need separate records.

## Interpret the metrics

- **L1 ↓:** mean absolute error; lower is better.
- **PSNR ↑:** higher is better. The implementation computes PSNR separately for each RGB channel, then averages the channels.
- **SSIM ↑:** higher is better; reported values are clamped to `[0, 1]`.

Metrics are computed per view and then averaged with equal weight across views in each group. Individual-object metrics use the full image area after applying the masks, rather than dividing by foreground area. Compare them using the same masks, image resolution, and view split. This evaluator does not compute LPIPS.

For the original GRAPHDECO feature benchmarks, see the [upstream 3DGS evaluation report](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/results.md).
