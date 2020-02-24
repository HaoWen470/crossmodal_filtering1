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
    parser.add_argument("--data_size", type=int, default=100)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--pretrain", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10, choices=[10, 100, 1000])
    parser.add_argument("--fusion_type", type=str, choices=["weighted", "poe", "sigma"], default="weighted")
    args = parser.parse_args()

    experiment_name = args.experiment_name
    dataset_args = {
        'use_proprioception': True,
        'use_haptics': True,
        'use_vision': True,
        'vision_interval': 2,
    }

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

    measurement = PandaEKFMeasurementModel()
    dynamics = PandaDynamicsModel(use_particles=False)
    ekf = KalmanFilterNetwork(dynamics, measurement)
    print("Creating model...")
    buddy = fannypack.utils.Buddy(experiment_name,
                                  ekf,
                                  optimizer_names=["ekf", "dynamics", "measurement"],
                                  load_checkpoint=True,
                                  )

    dataloader_dynamics = torch.utils.data.DataLoader(
        dynamics_recurrent_trainset, batch_size=args.batch, shuffle=True, num_workers=2, drop_last=True)

    for i in range(args.pretrain):
        print("Training dynamics epoch", i)
        training.train_dynamics_recurrent(buddy, ekf, dataloader_dynamics, optim_name="dynamics")
        print()

    buddy.save_checkpoint("phase_0_dynamics_pretrain")

    measurement_trainset_loader = torch.utils.data.DataLoader(
        dataset_measurement,
        batch_size=args.batch,
        shuffle=True,
        num_workers=16)

    for i in range(args.pretrain):
        print("Training measurement epoch", i)
        training.train_measurement(buddy, ekf, measurement_trainset_loader, log_interval=20, optim_name="measurement")
        print()

    buddy.save_checkpoint("phase_2_measurement_pretrain")

    e2e_trainset_loader = torch.utils.data.DataLoader(e2e_trainset, batch_size=args.batch, shuffle=True, num_workers=2)

    for i in range(args.epochs):
        if i < args.epochs / 2:
            obs_only = False
        else:
            obs_only = True
        print("Training ekf epoch", i)
        training.train_e2e(buddy, ekf, e2e_trainset_loader,
                           optim_name="ekf", obs_only=obs_only)

    buddy.save_checkpoint("phase_3_e2e")
