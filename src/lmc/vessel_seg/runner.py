"""Vessel segmentation runner — wraps the DIAS-trained 5-fold nnU-Net (clDice).

Loads `ckpt/vessel_seg_nnunet/{plans.json, dataset.json, fold_{0..4}/checkpoint_best.pth}`
and predicts a binary vessel mask for a single PNG frame using nnU-Net's
manual-initialised 5-fold ensemble (each fold's softmax probability map is
averaged inside `nnUNetPredictor.predict_from_files`).

Out-of-the-box assumes the trainer name `nnUNetTrainerClDice2D`. The
trainer module from `src/lmc/vessel_seg/trainer_cldice.py` must be
importable so nnU-Net can resolve the architecture during inference.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Optional, Sequence, Tuple

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


class VesselSegRunner:
    def __init__(
        self,
        model_dir: Path,
        folds: Sequence[int] = (0, 1, 2, 3, 4),
        checkpoint_name: str = "checkpoint_best.pth",
        trainer_name: str = "nnUNetTrainerClDice2D",
        configuration: str = "2d",
        device: Optional[torch.device] = None,
        prob_threshold: float = 0.5,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.folds = tuple(int(f) for f in folds)
        self.checkpoint_name = str(checkpoint_name)
        self.trainer_name = str(trainer_name)
        self.configuration = str(configuration)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.prob_threshold = float(prob_threshold)

        self._predictor = None

    def _build_predictor(self):
        nnUNetPredictor, nnUNetTrainer, PlansManager, determine_num_input_channels, load_json, join, isfile = _import_predictor()

        plans = load_json(join(str(self.model_dir), "plans.json"))
        dataset_json = load_json(join(str(self.model_dir), "dataset.json"))
        plans_manager = PlansManager(plans)
        configuration_manager = plans_manager.get_configuration(self.configuration)

        parameters = []
        inference_allowed_mirroring_axes = None
        for i, f in enumerate(self.folds):
            ckpt_path = join(str(self.model_dir), f"fold_{int(f)}", self.checkpoint_name)
            if not isfile(ckpt_path):
                raise FileNotFoundError(f"Missing vessel-seg checkpoint: {ckpt_path}")
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
        self._predictor = predictor
        return predictor

    @property
    def predictor(self):
        if self._predictor is None:
            self._build_predictor()
        return self._predictor

    def _stage_input(self, image: np.ndarray, work_dir: Path, caseid: str) -> Path:
        """nnU-Net 2D PNGs are read as 'imagesTr/<caseid>_0000.png' style."""
        in_dir = work_dir / "in"
        in_dir.mkdir(parents=True, exist_ok=True)
        if image.ndim == 2:
            arr = image
        elif image.ndim == 3 and image.shape[-1] in (1, 3):
            if image.shape[-1] == 3:
                arr = (0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]).astype(np.uint8)
            else:
                arr = image[..., 0]
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        Image.fromarray(arr).save(in_dir / f"{caseid}_0000.png")
        return in_dir

    def predict_mask(
        self,
        image: np.ndarray,
        caseid: str = "case",
        return_probability: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Run 5-fold ensemble vessel segmentation.

        Returns (binary_mask_uint8, prob_map_float32_or_None).
        """
        predictor = self.predictor
        with tempfile.TemporaryDirectory(prefix="lmc_vesselseg_") as tmp:
            tmp = Path(tmp)
            in_dir = self._stage_input(image, tmp, caseid)
            out_dir = tmp / "out"
            out_dir.mkdir(parents=True, exist_ok=True)
            predictor.predict_from_files(
                str(in_dir),
                str(out_dir),
                save_probabilities=return_probability,
                overwrite=True,
                num_processes_preprocessing=1,
                num_processes_segmentation_export=1,
            )
            seg_path = next(out_dir.glob(f"{caseid}.*"))
            seg = np.asarray(Image.open(seg_path).convert("L"), dtype=np.uint8)
            mask = (seg > 0).astype(np.uint8) * 255

            prob_map: Optional[np.ndarray] = None
            if return_probability:
                npz_path = out_dir / f"{caseid}.npz"
                if npz_path.exists():
                    with np.load(npz_path) as obj:
                        arr = obj.get("probabilities", None)
                        if arr is None and obj.files:
                            arr = obj[obj.files[0]]
                    if arr is not None:
                        arr = np.squeeze(np.asarray(arr))
                        if arr.ndim == 3 and arr.shape[0] >= 2:
                            prob_map = arr[1].astype(np.float32)
            shutil.rmtree(in_dir, ignore_errors=True)
            return mask, prob_map
