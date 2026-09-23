"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import numpy as np
import torch as th
import torch.distributed as dist
import argparse
import json
import os
import torch.nn as nn
from guided_diffusion import dist_util, logger
from guided_diffusion.axis_fusion import (
    FINAL_FUSION_MODES,
    FUSION_MODES,
    XSTART_FUSION_MODES,
)
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from dataloader_scripts.load_pet_2_5D import LoadTestData
import nibabel as nib
from scipy.ndimage import zoom


def resize_xy_to_shape(volume, target_shape):
    if volume.shape == target_shape:
        return volume
    if volume.shape[2] != target_shape[2]:
        volume = volume[:, :, :target_shape[2]]
    factors = (target_shape[0] / volume.shape[0], target_shape[1] / volume.shape[1], 1)
    return zoom(volume, factors, order=1)


def main():
    args = create_argparser().parse_args()
    if args.split not in ["train", "val", "test"]:
        raise ValueError("--split must be one of: train, val, test")
    if args.fusion_mode not in FUSION_MODES:
        raise ValueError(f"--fusion_mode must be one of: {', '.join(FUSION_MODES)}")
    if args.fusion_temperature <= 0:
        raise ValueError("--fusion_temperature must be positive")
    if not 0.0 <= args.final_prior_weight <= 1.0:
        raise ValueError("--final_prior_weight must be within [0, 1]")
    if args.save_axis_xstarts and args.fusion_mode not in XSTART_FUSION_MODES:
        raise ValueError("--save_axis_xstarts requires an xstart fusion mode")
    if args.save_axis_xstarts and args.final_prior_weight != 0:
        raise ValueError("--save_axis_xstarts requires --final_prior_weight 0 for exact final-axis provenance")
    if args.save_axis_final and args.fusion_mode not in FINAL_FUSION_MODES:
        raise ValueError("--save_axis_final requires final_mean or final_loo fusion")

    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(args.seed)

    # 处理 prior_start_t=None 的情况
    if args.prior_start_t == "None" or args.prior_start_t == None:
        args.prior_start_t = None

    if args.max_cases == "None" or args.max_cases == None:
        args.max_cases = None
    else:
        args.max_cases = int(args.max_cases)
    
    # Preserve the historical full-noise behavior unless an experiment
    # explicitly requests a matched no-learned-prior start timestep.
    if not args.use_prior and not args.allow_no_prior_start:
        args.prior_start_t = None

    args.in_channels = args.load_adj * 2 + 1 + args.out_channels

    dist_util.setup_dist()
    logger.configure()
    logger.log(
        f"fusion_mode={args.fusion_mode}, fusion_temperature={args.fusion_temperature}, seed={args.seed}"
    )
    logger.log("loading model and diffusion...")
    models = []
    diffusion = None
    for i in args.model_axis:
        model_path = os.path.join(args.model_root, f"model_{i}.pt")
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
        model.load_state_dict(
            dist_util.load_state_dict(model_path, map_location="cpu")
        )
        model.to(dist_util.dev())
        if args.use_fp16:
            model.convert_to_fp16()
        model.eval()
        model.requires_grad_(False)
        models.append(model)

    logger.log("sampling...")
    test_dir = os.path.join(args.data_root, args.split)
    requested_ids = []
    if args.case_id is not None:
        requested_ids.append(args.case_id)
    if getattr(args, "case_ids", None):
        requested_ids.extend(args.case_ids)
    only_ids = requested_ids if requested_ids else None
    test_input = LoadTestData(root_dir=test_dir, load_adj=args.load_adj, only_ids=only_ids)
    num_cases = len(test_input)
    if args.max_cases is not None and args.max_cases > 0:
        num_cases = min(num_cases, args.max_cases)
    case_indices = list(range(num_cases))
    if args.case_id is not None and not getattr(args, "case_ids", None):
        requested_id = args.case_id.rstrip("_")
        matches = [
            case_idx
            for case_idx in range(len(test_input))
            if test_input.get_name(case_idx).rstrip("_") == requested_id
        ]
        if len(matches) != 1:
            raise ValueError(f"--case_id {args.case_id!r} matched {len(matches)} cases")
        case_indices = matches
    prior_root = args.load_prior_root
    if args.use_prior:
        split_prior_root = os.path.join(args.load_prior_root, args.split)
        if os.path.isdir(split_prior_root):
            prior_root = split_prior_root
        elif args.allow_shared_prior_root:
            logger.log(
                f"using shared prior root for split={args.split}: {args.load_prior_root}"
            )
        elif args.split != "test":
            raise FileNotFoundError(
                f"split={args.split} with --use_prior requires split-specific prior files in "
                f"{split_prior_root}. This avoids accidentally using test priors with train/val IDs."
            )

    for idx in case_indices:
        print(f"idx: {idx} / {len(test_input)}")
        whole_image = None
        sample_fn = diffusion.p_sample_loop
        model_kwargs = {}
        test_input.idx = idx
        
        shape = (args.image_size, args.image_size, test_input.get_zsize())
        comb_img = np.zeros(shape)
        axis_xstart_sum = None
        axis_final_sum = None
        
        # 处理先验数据 (兼容 "0000_umap_pred.nii" 与 "0000umap_pred.nii" 两种命名)
        patient_id = test_input.get_name(idx)
        prior_path = os.path.join(prior_root, f"{patient_id}umap_pred.nii")
        if not os.path.exists(prior_path):
            prior_path_alt = os.path.join(prior_root, f"{patient_id.rstrip('_')}umap_pred.nii")
            if os.path.exists(prior_path_alt):
                prior_path = prior_path_alt
            else:
                prior_path = None
        if args.use_prior and prior_path is not None:
            prior_nii = nib.load(prior_path)
            prior_numpy = prior_nii.get_fdata()
        else:
            prior_numpy = None
        
        for i in range(args.sample_num):
            prior = th.zeros(shape).to(dist_util.dev())
            if prior_numpy is not None:
                if prior_numpy.shape != shape:
                    prior_numpy_model = resize_xy_to_shape(prior_numpy, shape).astype(np.float32)
                else:
                    prior_numpy_model = prior_numpy.astype(np.float32)
                prior[:, :, :test_input.get_original_z()] = th.from_numpy(prior_numpy_model[:, :, :test_input.get_original_z()]).to(dist_util.dev())
            
            noisy_priors = []
            for n in range(args.avg_start_number): 
                if args.prior_start_t is not None and args.prior_start_t < diffusion.num_timesteps:
                    noisy_priors.append(diffusion.q_sample(prior, th.tensor(args.prior_start_t).to(dist_util.dev())))
                else:
                    noisy_priors.append(th.randn(shape).to(dist_util.dev()))
            sampling_start = (
                None
                if args.prior_start_t is not None and args.prior_start_t >= diffusion.num_timesteps
                else args.prior_start_t
            )
            fusion_weight_history = []
            final_axis_xstarts = None
            final_axis_outputs = None

            def record_fusion_weights(timestep, weights):
                spatial_dims = tuple(range(1, weights.ndim))
                fusion_weight_history.append(
                    (int(timestep), weights.mean(dim=spatial_dims).detach().cpu().tolist())
                )

            def record_final_axis_xstarts(timestep, axis_xstarts):
                nonlocal final_axis_xstarts
                if int(timestep) == 0:
                    final_axis_xstarts = [value.detach().cpu().numpy().copy() for value in axis_xstarts]

            def record_final_axis_outputs(axis_outputs):
                nonlocal final_axis_outputs
                final_axis_outputs = [
                    value.detach().cpu().numpy().copy() for value in axis_outputs
                ]

            whole_image = sample_fn(
                models,
                args.model_axis,
                test_input,
                shape,
                args.batch_size,
                sampling_start,
                noise=noisy_priors,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                fusion_mode=args.fusion_mode,
                fusion_temperature=args.fusion_temperature,
                fusion_weight_callback=record_fusion_weights if args.save_fusion_stats else None,
                axis_xstart_callback=record_final_axis_xstarts if args.save_axis_xstarts else None,
                axis_final_callback=record_final_axis_outputs if args.save_axis_final else None,
            )
            if args.save_axis_xstarts:
                if final_axis_xstarts is None:
                    raise RuntimeError("sampler did not expose final axis x_start predictions")
                if axis_xstart_sum is None:
                    axis_xstart_sum = [np.zeros_like(value, dtype=np.float64) for value in final_axis_xstarts]
                for axis_index, value in enumerate(final_axis_xstarts):
                    axis_xstart_sum[axis_index] += value
            if args.save_axis_final:
                if final_axis_outputs is None:
                    raise RuntimeError("sampler did not expose final axis outputs")
                if axis_final_sum is None:
                    axis_final_sum = [
                        np.zeros_like(value, dtype=np.float64)
                        for value in final_axis_outputs
                    ]
                for axis_index, value in enumerate(final_axis_outputs):
                    axis_final_sum[axis_index] += value
            whole_image = whole_image.cpu().numpy()
            if args.final_prior_weight > 0 and prior_numpy is not None:
                prior_volume = prior.cpu().numpy()
                whole_image = (
                    (1.0 - args.final_prior_weight) * whole_image
                    + args.final_prior_weight * prior_volume
                )
            comb_img += whole_image
            whole_image = whole_image[:,:,:test_input.get_original_z()]
            whole_image = nib.Nifti1Image(whole_image, affine=np.eye(4))

            axis = "".join(args.model_axis)

            if args.save_single:
                # Save individual sampled volume
                output_dir = os.path.join(args.save_root, f"adj{args.load_adj}_models_{axis}", f"noise_{args.avg_start_number}_priort_{args.prior_start_t}_ave_first_ddpm_full_single")
                os.makedirs(output_dir, exist_ok=True)

                save_path = os.path.join(output_dir, f"{test_input.get_name(idx)}pred_{i}.nii")
                nib.save(whole_image, save_path)

            if args.save_fusion_stats:
                fusion_stats_dir = os.path.join(args.save_root, f"adj{args.load_adj}_models_{axis}", f"noise_{args.avg_start_number}_priort_{args.prior_start_t}_fusion_stats")
                os.makedirs(fusion_stats_dir, exist_ok=True)
                stats_path = os.path.join(fusion_stats_dir, f"{test_input.get_name(idx)}pred_{i}.json")
                with open(stats_path, "w", encoding="utf-8") as stats_file:
                    json.dump(
                        {
                            "fusion_mode": args.fusion_mode,
                            "fusion_temperature": args.fusion_temperature,
                            "axis_order": args.model_axis,
                            "seed": args.seed,
                            "final_prior_weight": args.final_prior_weight,
                            "weights_by_timestep": fusion_weight_history,
                        },
                        stats_file,
                    )

        # Save averaged combined volume
        comb_img /= args.sample_num
        comb_img = comb_img[:, :, :test_input.get_original_z()]
        comb_img = resize_xy_to_shape(comb_img, test_input.get_original_shape()).astype(np.float32)
        comb_img[comb_img < 0] = 0
        comb_img = nib.Nifti1Image(comb_img, affine=test_input.get_affine())

        output_dir_comb = os.path.join(args.save_root, f"adj{args.load_adj}_models_{axis}", f"noise_{args.avg_start_number}_priort_{args.prior_start_t}_comb")
        os.makedirs(output_dir_comb, exist_ok=True)

        save_path_comb = os.path.join(output_dir_comb, f"{test_input.get_name(idx)}pred.nii")
        nib.save(comb_img, save_path_comb)
        if args.save_axis_xstarts:
            axis_output_dir = os.path.join(args.save_root, f"adj{args.load_adj}_models_{axis}", f"noise_{args.avg_start_number}_priort_{args.prior_start_t}_axis_xstarts")
            os.makedirs(axis_output_dir, exist_ok=True)
            for axis_name, value in zip(args.model_axis, axis_xstart_sum):
                value = (value / args.sample_num)[:, :, :test_input.get_original_z()]
                value = resize_xy_to_shape(value, test_input.get_original_shape()).astype(np.float32)
                axis_path = os.path.join(axis_output_dir, f"{test_input.get_name(idx)}{axis_name}_xstart.nii")
                nib.save(nib.Nifti1Image(value, affine=test_input.get_affine()), axis_path)
        if args.save_axis_final:
            axis_output_dir = os.path.join(
                args.save_root,
                f"adj{args.load_adj}_models_{axis}",
                f"noise_{args.avg_start_number}_priort_{args.prior_start_t}_axis_final",
            )
            os.makedirs(axis_output_dir, exist_ok=True)
            for axis_name, value in zip(args.model_axis, axis_final_sum):
                value = (value / args.sample_num)[:, :, :test_input.get_original_z()]
                value = resize_xy_to_shape(
                    value, test_input.get_original_shape()
                ).astype(np.float32)
                axis_path = os.path.join(
                    axis_output_dir,
                    f"{test_input.get_name(idx)}{axis_name}_final.nii",
                )
                nib.save(nib.Nifti1Image(value, affine=test_input.get_affine()), axis_path)




def create_argparser():
    defaults = dict(
        clip_denoised=True,
        batch_size=64,
        use_ddim=False,
        out_channels=1,
        model_root="weights/axis_models",
        model_axis=["x", "y", "z"],
        prior_start_t=200, #range from 1 to 999 for starting noise level add to prior, None for no prior
        load_adj=8,
        avg_start_number=2,
        sample_num=1,
        save_root="outputs/samples/run",
        load_prior_root="outputs/priors/refined",
        data_root="data/udpet",
        split="test",
        save_single=True,
        max_cases=None,
        case_id=None,
        use_prior=False,  # 是否使用先验数据
        allow_no_prior_start=False,
        allow_shared_prior_root=False,
        final_prior_weight=0.0,
        fusion_mode="mean",  # mean reproduces MADM; research variants are agreement/xstart_agreement
        fusion_temperature=0.05,
        save_fusion_stats=True,
        save_axis_xstarts=False,
        save_axis_final=False,
        seed=20260819,
    )
    defaults.update(model_and_diffusion_defaults())
    model_axis = defaults.pop("model_axis")

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.add_argument("--model_axis", nargs="+", default=model_axis)
    parser.add_argument("--case_ids", nargs="+", default=None,
                        help="sorted case ids to sample in one process, e.g. --case_ids 0000 0001")
    return parser


if __name__ == "__main__":
    main()
