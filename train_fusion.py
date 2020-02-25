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
    parser.add_argument("--train", type=str, choices=[ "all", "fusion", "ekf"], default="all")
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--module_type", type=str, default="all", choices=["all", "ekf"])
    parser.add_argument("--blackout", type=float, default=0.0)
    parser.add_argument("--mass", action="store_true")
    args = parser.parse_args()

    experiment_name = args.experiment_name
    dataset_args = {
        'use_proprioception': True,
        'use_haptics': True,
        'use_vision': True,
        'vision_interval': 2,
        'image_blackout_ratio': args.blackout,
        'use_mass': args.mass,
    }
    # image_modality_model
    image_measurement = PandaEKFMeasurementModel(missing_modalities=['gripper_sensors'])
    image_dynamics = PandaDynamicsModel(use_particles=False)
    image_model = KalmanFilterNetwork(image_dynamics, image_measurement)

    # force_modality_model
    force_measurement = PandaEKFMeasurementModel(missing_modalities=['image'])
    force_dynamics = PandaDynamicsModel(use_particles=False)
    force_model = KalmanFilterNetwork(force_dynamics, force_measurement)

    #weight model and fusion model
    weight_model = CrossModalWeights()
    fusion_model = KalmanFusionModel(image_model, force_model, weight_model, fusion_type=args.fusion_type)

    buddy = fannypack.utils.Buddy(experiment_name,
                                  fusion_model,
                                  optimizer_names=["im_meas", "force_meas",
                                                   "dynamics", "dynamics_recurr",
                                                   "force_ekf", "im_ekf",
                                                   "fusion"],
                                  load_checkpoint=True,
                                  )

    if args.load_checkpoint is not None:
        if args.module_type == "all":
            buddy.load_checkpoint(path = args.load_checkpoint)
        if args.module_type == "ekf":
            buddy.load_checkpoint_module(source="image_model", path=args.load_checkpoint)
            buddy.load_checkpoint_module(source="force_model", path=args.load_checkpoint)

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

    dataset_dynamics = panda_datasets.PandaDynamicsDataset(
        'data/gentle_push_{}.hdf5'.format(args.data_size),
        subsequence_length=16,
        **dataset_args)

    #train everything
    if args.train == "all":
        # training dynamics model

        dataloader_dynamics = torch.utils.data.DataLoader(
            dataset_dynamics, batch_size=args.batch, shuffle=True, num_workers=2, drop_last=True)

        for i in range(args.pretrain):
            print("Training dynamics epoch", i)
            training.train_dynamics(buddy, image_model,
                                    dataloader_dynamics, optim_name="dynamics")
            print()

        buddy.save_checkpoint("phase_0_dynamics_pretrain")

        # recurrence pretrain
        dataloader_dynamics_recurr = torch.utils.data.DataLoader(
            dynamics_recurrent_trainset, batch_size=args.batch, shuffle=True, num_workers=2, drop_last=True)

        for i in range(args.pretrain):
            print("Training recurr dynamics epoch", i)
            training.train_dynamics_recurrent(buddy, image_model,
                                              dataloader_dynamics_recurr, optim_name="dynamics_recurr")
            print()


        buddy.save_checkpoint("phase_1_dynamics_recurrent_pretrain")
        #load force model from image model dynamics
        buddy.load_checkpoint_module(source="image_model.dynamics_model",
                                     target="force_model.dynamics_model",
                                     label="phase_1_dynamics_recurrent_pretrain")

        measurement_trainset_loader = torch.utils.data.DataLoader(
            dataset_measurement,
            batch_size=args.batch*2,
            shuffle=True,
            num_workers=8)

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


    e2e_trainset_loader = torch.utils.data.DataLoader(e2e_trainset, batch_size=args.batch,
                                                      shuffle=True, num_workers=2)
    #train ekf (or all)
    image_model.freeze_dynamics_model = True
    force_model.freeze_dynamics_model = True
    image_model.freeze_measurement_model = False
    force_model.freeze_measurement_model = False

    if args.train == "all" or args.train == "ekf":
        for i in range(args.pretrain):
            print("Training ekf epoch", i)
            obs_only=False
            training.train_e2e(buddy, force_model,
                               e2e_trainset_loader, optim_name="force_ekf", obs_only=obs_only)
            training.train_e2e(buddy, image_model, e2e_trainset_loader, optim_name="im_ekf")

        buddy.save_checkpoint("phase_3_e2e")

    # train fusion
    for i in range(args.epochs):
        print("Training fusion epoch", i)
        obs_only=False
        training.train_fusion(buddy, fusion_model, e2e_trainset_loader,
                              optim_name="fusion", obs_only=obs_only)

    buddy.save_checkpoint("phase_4_fusion")