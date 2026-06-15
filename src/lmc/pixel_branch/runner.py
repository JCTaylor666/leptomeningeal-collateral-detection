"""Pixel-branch runner — wraps the Zurich-collateral 5-fold nnU-Net (PRAUC).

Reads `ckpt/pixel_branch_nnunet/{plans.json, dataset.json, fold_{0..4}/checkpoint_best_prauc.pth}`.

Two modes:
- ensemble: average soft-probabilities across the requested folds (default
  for new images).
- fold: use a single fold checkpoint (used for OOF regression where each
  caseid is paired with the fold whose val set contains it).

The trainer module `src/lmc/pixel_branch/trainer_prauc.py` must be
importable so nnU-Net can resolve the trainer architecture during
inference.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
from PIL import Image


def _import_predictor():
    from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile  # noqa: F401
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    return nnUNetPredictor, nnUNetTrainer, PlansManager, determine_num_input_channels, load_json, join, isfile


class PixelBranchRunner:
    def __init__(
        self,
        model_dir: Path,
        checkpoint_name: str = "checkpoint_best_prauc.pth",
        trainer_name: str = "nnUNetTrainerPRAUC2D",
        configuration: str = "2d",
        device: Optional[torch.device] = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.checkpoint_name = str(checkpoint_name)
        self.trainer_name = str(trainer_name)
        self.configuration = str(configuration)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._predictors: dict = {}

    def _build_predictor(self, folds: Sequence[int]):
        key = tuple(sorted(int(f) for f in folds))
        if key in self._predictors:
            return self._predictors[key]
        nnUNetPredictor, nnUNetTrainer, PlansManager, determine_num_input_channels, load_json, join, isfile = _import_predictor()

        plans = load_json(join(str(self.model_dir), "plans.json"))
        dataset_json = load_json(join(str(self.model_dir), "dataset.json"))
        plans_manager = PlansManager(plans)
        configuration_manager = plans_manager.get_configuration(self.configuration)

        parameters = []
        inference_allowed_mirroring_axes = None
        for i, f in enumerate(key):
            ckpt_path = join(str(self.model_dir), f"fold_{int(f)}", self.checkpoint_name)
            if not isfile(ckpt_path):
                raise FileNotFoundError(f"Missing pixel-branch checkpoint: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"), weights_only=False)
            if i == 0:
                inference_allowed_mirroring_axes = checkpoint.get("inference_allowed_mirroring_axes", None)
            parameters.append(checkpoint["network_weights"])

        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
        network = nnUNetTrainer.build_network_architecture(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
            enable_deep_supervision=False,
        )
        network.load_state_dict(parameters[0])

        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=self.device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )
        predictor.manual_initialization(
            network=network,
            plans_manager=plans_manager,
            configuration_manager=configuration_manager,
            parameters=parameters,
            dataset_json=dataset_json,
            trainer_name=self.trainer_name,
            inference_allowed_mirroring_axes=inference_allowed_mirroring_axes,
        )
        self._predictors[key] = predictor
        return predictor

    @staticmethod
    def _stage_input(
        image: np.ndarray,
        work_dir: Path,
        caseid: str,
        vessel_mask: Optional[np.ndarray] = None,
    ) -> Path:
        in_dir = work_dir / "in"
        in_dir.mkdir(parents=True, exist_ok=True)
        if image.ndim == 2:
            arr = image
        elif image.ndim == 3 and image.shape[-1] in (1, 3):
            arr = image[..., 0] if image.shape[-1] == 1 else (
                0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
            ).astype(np.uint8)
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        # Vessel-mask gating: the pixel branch is trained on vessel-masked input
        # (Dataset708 = image * (vessel_mask > 0); see pixel_branch/prepare_zurich.py).
        # The same gating must be applied at inference, otherwise the masked-trained
        # nnU-Net sees out-of-distribution (full-frame) input and returns near-zero
        # probabilities, which would silently neutralise the pixel branch in fusion.
        if vessel_mask is not None:
            vm = (np.asarray(vessel_mask) > 0).astype(np.uint8)
            if vm.shape != arr.shape:
                raise ValueError(
                    f"vessel_mask shape {vm.shape} does not match image shape {arr.shape}"
                )
            arr = (arr * vm).astype(np.uint8)
        Image.fromarray(arr).save(in_dir / f"{caseid}_0000.png")
        return in_dir

    def predict_prob(
        self,
        image: np.ndarray,
        caseid: str = "case",
        folds: Iterable[int] = (0, 1, 2, 3, 4),
        vessel_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return per-pixel collateral probability (foreground class).

        When ``vessel_mask`` is provided, the input image is gated by it
        (image * (vessel_mask > 0)) to match the masked training data
        (Dataset708). This is required for the shipped pixel-branch
        checkpoints to produce meaningful probabilities; omitting it yields
        near-zero output for every pixel.
        """
        predictor = self._build_predictor(list(folds))
        with tempfile.TemporaryDirectory(prefix="lmc_pixelbr_") as tmp:
            tmp = Path(tmp)
            in_dir = self._stage_input(image, tmp, caseid, vessel_mask=vessel_mask)
            out_dir = tmp / "out"
            out_dir.mkdir(parents=True, exist_ok=True)
            predictor.predict_from_files(
                str(in_dir),
                str(out_dir),
                save_probabilities=True,
                overwrite=True,
                num_processes_preprocessing=1,
                num_processes_segmentation_export=1,
            )
            npz_path = out_dir / f"{caseid}.npz"
            if not npz_path.exists():
                raise RuntimeError(f"nnU-Net did not produce {npz_path}")
            with np.load(npz_path) as obj:
                arr = obj.get("probabilities", None)
                if arr is None and obj.files:
                    arr = obj[obj.files[0]]
            if arr is None:
                raise RuntimeError(f"No probability array in {npz_path}")
            arr = np.squeeze(np.asarray(arr))
            if arr.ndim == 3 and arr.shape[0] >= 2:
                prob = arr[1].astype(np.float32)
            elif arr.ndim == 2:
                prob = arr.astype(np.float32)
            else:
                raise RuntimeError(f"Unexpected probability shape: {arr.shape}")
            shutil.rmtree(in_dir, ignore_errors=True)
            return np.clip(prob, 0.0, 1.0)


def pool_pixel_prob_to_nodes(
    prob_map: np.ndarray,
    nodes: Sequence[dict],
) -> np.ndarray:
    """Mean-pool pixel probability over each node's pixel set (paper Eq. 2)."""
    height, width = prob_map.shape
    out = np.zeros(len(nodes), dtype=np.float32)
    for node_idx, node in enumerate(nodes):
        pix = node.get("pixels", [])
        if not pix:
            continue
        pix_arr = np.asarray(pix, dtype=np.int64)
        if pix_arr.ndim != 2 or pix_arr.shape[1] < 2:
            continue
        ys = np.clip(pix_arr[:, 0], 0, height - 1)
        xs = np.clip(pix_arr[:, 1], 0, width - 1)
        out[node_idx] = float(np.mean(prob_map[ys, xs]))
    return out
