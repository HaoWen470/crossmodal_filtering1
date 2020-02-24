import torch
import fannypack
from lib import panda_datasets
from lib.ekf import KalmanFilterNetwork
from fannypack import utils
from lib import dpf
from lib.panda_models import PandaDynamicsModel, PandaEKFMeasurementModel

from lib.fusion import KalmanFusionModel
from lib.fusion import CrossModalWeights

import lib.panda_kf_training as training
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="fusion",
    )
    parser.add_argument("--data_size", type=int, default=100, choices=[10, 100, 1000])
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--pretrain", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fusion_type", type=str, choices=["weighted", "poe", "sigma"], default="weighted")
    parser.add_argument("--train", type=str, choices=[ "all", "fusion"], default="all")

    args = parser.parse_args()

    experiment_name = args.experiment_name
    dataset_args = {
        'use_proprioception': True,
        'use_haptics': True,
        'use_vision': True,
        'vision_interval': 2,
    }
    # image_modality_model
    image_measurement = PandaEKFMeasurementModel(missing_modalities=['gripper_sensors'])
    image_dynamics = PandaDynamicsModel(use_particles=False)
    image_model = KalmanFilterNetwork(image_dynamics, image_measurement)

    # force_modality_model
    force_measurement = PandaEKFMeasurementModel(missing_modalities=['image'])
    force_dynamics = PandaDynamicsModel(use_particles=False)
    force_model = KalmanFilterNetwork(force_dynamics, force_measurement)

    weight_model = CrossModalWeights()

    fusion_model = KalmanFusionModel(image_model, force_model, weight_model, fusion_type=args.fusion_type)

    models = {'image': image_model, 'force': force_model, 'weight': weight_model}
    # todo: need a different version of buddy... also probably need to load and save myself
    buddy = fannypack.utils.Buddy(experiment_name,
                                  fusion_model,
                                  optimizer_names=["im_meas", "force_meas",
                                                   "im_dynamics", "force_dynamics",
                                                   "force_ekf", "im_ekf",
                                                   "fusion"],
                                  load_checkpoint=True,
                                  )
    print("Creating dataset...")
    # dataset_full = panda_datasets.PandaParticleFilterDataset(
    #     'data/gentle_push_10.hdf5',
    #     subsequence_length=16,
    #     **dataset_args)

    e2e_trainset = panda_datasets.PandaParticleFilterDataset(
        "data/gentle_push_{}.hdf5".format(args.data_size),
        subsequence_length=16,
        particle_count=1,
        particle_stddev=(.03, .03),
        **dataset_args
    )

    dataset_measurement = panda_datasets.PandaMeasurementDataset(
        "data/gentle_push_{}.hdf5".format(args.data_size),
        subsequence_length=16,
        stddev=(0.5, 0.5),
        samples_per_pair=20,
        **dataset_args)

    dynamics_recurrent_trainset = panda_datasets.PandaSubsequenceDataset(
        "data/gentle_push_{}.hdf5".format(args.data_size),
        subsequence_length=32,
        **dataset_args
    )
    if args.train == "all":
        dataloader_dynamics = torch.utils.data.DataLoader(
            dynamics_recurrent_trainset, batch_size=args.batch, shuffle=True, num_workers=2, drop_last=True)

        for i in range(args.pretrain):
            print("Training dynamics epoch", i)
            training.train_dynamics_recurrent(buddy, image_model, dataloader_dynamics, optim_name="im_dynamics")
            training.train_dynamics_recurrent(buddy, force_model, dataloader_dynamics, optim_name="force_dynamics")
            print()

        buddy.save_checkpoint("phase_0_dynamics_pretrain")

        measurement_trainset_loader = torch.utils.data.DataLoader(
            dataset_measurement,
            batch_size=args.batch*2,
            shuffle=True,
            num_workers=16)

        for i in range(int(args.pretrain/2)):
            print("Training measurement epoch", i)
            training.train_measurement(buddy, image_model, measurement_trainset_loader,
                                       log_interval=20, optim_name="im_meas",
                                       checkpoint_interval= args.data_size*10)
            training.train_measurement(buddy, force_model, measurement_trainset_loader,
                                       log_interval=20, optim_name="force_meas",
                                       checkpoint_interval= args.data_size*10)
            print()

        buddy.save_checkpoint("phase_2_measurement_pretrain")

        e2e_trainset_loader = torch.utils.data.DataLoader(e2e_trainset, batch_size=args.batch, shuffle=True, num_workers=2)

        for i in range(args.pretrain):
            print("Training ekf epoch", i)
            if i < args.pretrain/2:
                obs_only=False
            else:
                obs_only=True
            training.train_e2e(buddy, force_model,
                               e2e_trainset_loader, optim_name="force_ekf", obs_only=obs_only)
            training.train_e2e(buddy, image_model, e2e_trainset_loader, optim_name="im_ekf")

        buddy.save_checkpoint("phase_3_e2e")

    e2e_trainset_loader = torch.utils.data.DataLoader(e2e_trainset, batch_size=args.batch, shuffle=True, num_workers=2)

    for i in range(args.epochs):
        print("Training fusion epoch", i)
        if i < args.epochs/2:
            obs_only=False
        else:
            obs_only=True
        training.train_fusion(buddy, fusion_model, e2e_trainset_loader,
                              optim_name="fusion", obs_only=obs_only)

    buddy.save_checkpoint("phase_4_fusion")