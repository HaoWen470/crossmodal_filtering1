import torch
import fannypack
from lib import panda_datasets, omnipush_datasets
from lib.ekf import KalmanFilterNetwork
from fannypack import utils
from lib import dpf
from lib.panda_models import PandaDynamicsModel, PandaEKFMeasurementModel2GAP

from lib.fusion import KalmanFusionModel
from lib.fusion import CrossModalWeights

import lib.panda_kf_training as training
import argparse
import gc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n',
                        "--experiment_name",
                        type=str,
                        default="fusion",
                        )
    parser.add_argument("--data_size", type=int, default=1000, choices=[10, 100, 1000])
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--pretrain", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--fusion_type", type=str, choices=["cross", "uni"], default="cross")
    parser.add_argument("--train", type=str, choices=["all", "fusion", "ekf"], default="all")
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--module_type", type=str, default="all", choices=["all", "ekf"])
    parser.add_argument("--blackout", type=float, default=0.0)
    parser.add_argument("--mass", action="store_true")
    parser.add_argument("--omnipush", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden_units", type=int, default=64)
    parser.add_argument("--many_loss", action="store_true")
    parser.add_argument("--init_state_noise", type=float, default=0.2)
    parser.add_argument("--sequential_image", type=int, default=1)
    parser.add_argument("--start_timestep", type=int, default=0)
    parser.add_argument("--no_proprio", action="store_true")
    parser.add_argument("--learnable_Q", action="store_true")
    parser.add_argument("--obs_only", action="store_true")
    parser.add_argument("--meas_loss", choices=['mse', 'nll', 'mixed'], default="mse")
    parser.add_argument("--ekf_loss", choices=['mse', 'nll', 'mixed'], default="mse")
    parser.add_argument("--meas_lr", default=1e-5, type=float)
    parser.add_argument("--ekf_lr", default=5e-6, type=float)
    parser.add_argument("--freeze_dyn", action="store_true")

    args = parser.parse_args()

    experiment_name = args.experiment_name
    dataset_args = {
        'use_proprioception': not args.no_proprio,
        'use_haptics': True,
        'use_vision': True,
        'vision_interval': 2,
        'image_blackout_ratio': args.blackout,
        'use_mass': args.mass,
        'hidden_units': args.hidden_units,
        'batch': args.batch,
        'pretrain epochs': args.pretrain,
        'omnipush dataset': args.omnipush,
        'start training from': args.train,
        'epochs': args.epochs,
        'loading checkpoint': args.load_checkpoint,
        'init state noise': args.init_state_noise,
        'many loss': args.many_loss,
        'sequential_image_rate': args.sequential_image,
        'start_timestep': args.start_timestep,
        'measurement_loss': args.meas_loss,
        'ekf_loss': args.ekf_loss,
        'learnable_Q': args.learnable_Q,
        'obs_only': args.obs_only,
        'freeze_dyn': args.freeze_dyn,
        'meas_lr': args.meas_lr,
        'ekf_lr': args.ekf_lr,

    }
    # image_modality_model

    image_measurement = PandaEKFMeasurementModel2GAP(missing_modalities=['gripper_sensors'],
                                                     units=args.hidden_units, )
    image_dynamics = PandaDynamicsModel(use_particles=False, learnable_Q=args.learnable_Q)
    image_model = KalmanFilterNetwork(image_dynamics, image_measurement)

    # force_modality_model
    force_measurement = PandaEKFMeasurementModel2GAP(missing_modalities=['image'],
                                                     units=args.hidden_units,
                                                     )
    force_dynamics = PandaDynamicsModel(use_particles=False,
                                        learnable_Q=args.learnable_Q)
    force_model = KalmanFilterNetwork(force_dynamics, force_measurement)

    # weight model and fusion model
    weight_model = CrossModalWeights(state_dim=2)

    if args.sequential_image > 1:
        know_image_blackout = True
        print("blackout")
    else:
        know_image_blackout = False

    fusion_model = KalmanFusionModel(image_model, force_model, weight_model,
                                     fusion_type=args.fusion_type,
                                     know_image_blackout=know_image_blackout)

    buddy = fannypack.utils.Buddy(experiment_name,
                                  fusion_model,
                                  optimizer_names=["im_meas", "force_meas",
                                                   "dynamics", "dynamics_recurr",
                                                   "force_ekf", "im_ekf",
                                                   "fusion"],
                                  )
    buddy.add_metadata(dataset_args)

    if args.load_checkpoint is not None:
        if args.module_type == "all":
            buddy.load_checkpoint(path=args.load_checkpoint)
        if args.module_type == "ekf":
            buddy.load_checkpoint_module(source="image_model",
                                         path=args.load_checkpoint)
            buddy.load_checkpoint_module(source="force_model",
                                         path=args.load_checkpoint)

    print("Creating dataset...")

    if args.omnipush:
        e2e_trainset = omnipush_datasets.OmnipushParticleFilterDataset(
            "simpler/train0.hdf5",
            "simpler/train1.hdf5",
            "simpler/train2.hdf5",
            "simpler/train3.hdf5",
            "simpler/train4.hdf5",
            "simpler/train5.hdf5",
            subsequence_length=16,
            particle_count=1,
            particle_stddev=(.03, .03),
            **dataset_args
        )

        dataset_measurement = omnipush_datasets.OmnipushMeasurementDataset(
            "simpler/train0.hdf5",
            "simpler/train1.hdf5",
            "simpler/train2.hdf5",
            "simpler/train3.hdf5",
            "simpler/train4.hdf5",
            "simpler/train5.hdf5",
            subsequence_length=16,
            stddev=(0.5, 0.5),
            samples_per_pair=20,
            **dataset_args)

        dynamics_recurrent_trainset = omnipush_datasets.OmnipushSubsequenceDataset(
            "simpler/train0.hdf5",
            "simpler/train1.hdf5",
            "simpler/train2.hdf5",
            "simpler/train3.hdf5",
            "simpler/train4.hdf5",
            "simpler/train5.hdf5",
            subsequence_length=32,
            **dataset_args
        )

        dataset_dynamics = omnipush_datasets.OmnipushDynamicsDataset(
            "simpler/train0.hdf5",
            "simpler/train1.hdf5",
            "simpler/train2.hdf5",
            "simpler/train3.hdf5",
            "simpler/train4.hdf5",
            "simpler/train5.hdf5",
            subsequence_length=16,
            **dataset_args)
    else:
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

    # train everything
    if args.train == "all":
        # training dynamics model
        dataloader_dynamics = torch.utils.data.DataLoader(
            dataset_dynamics, batch_size=args.batch, shuffle=True, num_workers=2, drop_last=True)

        for i in range(args.pretrain):
            print("Training dynamics epoch", i)
            training.train_dynamics(buddy, image_model,
                                    dataloader_dynamics, optim_name="dynamics")
            print()

        buddy.save_checkpoint("temp_1")
        buddy.load_checkpoint_module(source="image_model.dynamics_model",
                                     target="force_model.dynamics_model",
                                     label="temp_1")
        buddy.save_checkpoint("phase_0_dynamics_pretrain")

        # recurrence pretrain
        dataloader_dynamics_recurr = torch.utils.data.DataLoader(
            dynamics_recurrent_trainset, batch_size=args.batch, shuffle=True, num_workers=2, drop_last=True)

        for i in range(args.pretrain):
            print("Training recurr dynamics epoch", i)
            training.train_dynamics_recurrent(buddy, image_model,
                                              dataloader_dynamics_recurr, optim_name="dynamics_recurr")
            print()

        buddy.save_checkpoint("temp_2")
        buddy.load_checkpoint_module(source="image_model.dynamics_model",
                                     target="force_model.dynamics_model",
                                     label="temp_2")
        buddy.save_checkpoint('phase_1_dynamics_recurrent_pretrain')

        # training measurement model
        measurement_trainset_loader = torch.utils.data.DataLoader(
            dataset_measurement,
            batch_size=args.batch,
            shuffle=True,
            num_workers=8)

        buddy.set_learning_rate(args.meas_lr,
                                optimizer_name="force_meas")

        buddy.set_learning_rate(args.meas_lr,
                                optimizer_name="im_meas")

        for i in range(int(args.pretrain)):
            print("Training img measurement epoch", i)
            training.train_measurement(buddy, image_model,
                                       measurement_trainset_loader,
                                       log_interval=20, optim_name="im_meas",
                                       checkpoint_interval=10000,
                                       loss_type=args.meas_loss)
            print()
        for i in range(int(args.pretrain)):
            print("Training force measurement epoch", i)

            training.train_measurement(buddy, force_model,
                                       measurement_trainset_loader,
                                       log_interval=20, optim_name="force_meas",
                                       checkpoint_interval=10000,
                                       loss_type=args.meas_loss)

        buddy.save_checkpoint("phase_2_measurement_pretrain")

    e2e_trainset_loader = torch.utils.data.DataLoader(e2e_trainset,
                                                      batch_size=args.batch,
                                                      shuffle=True,
                                                      num_workers=2)

    # train e2e ekf
    if args.train == "all" or args.train == "ekf":

        buddy.set_learning_rate(args.ekf_lr,
                                optimizer_name="force_ekf")
        buddy.set_learning_rate(args.ekf_lr,
                                optimizer_name="im_ekf")
        if args.freeze_dyn:
            fannypack.utils.freeze_module(image_model.dynamics_model)
            fannypack.utils.freeze_module(force_model.dynamics_model)

        for i in range(args.pretrain):
            print("Training force ekf epoch", i)
            training.train_e2e(buddy, force_model,
                               e2e_trainset_loader,
                               optim_name="force_ekf",
                               loss_type=args.ekf_loss)

        for i in range(args.pretrain):
            print("Training img ekf epoch", i)
            training.train_e2e(buddy, image_model,
                               e2e_trainset_loader,
                               optim_name="im_ekf",
                               loss_type=args.ekf_loss)

        buddy.save_checkpoint("phase_3_e2e")

    # train fusion
    buddy.set_learning_rate(args.lr, optimizer_name="fusion")

    for i in range(args.epochs):
        print("Training fusion epoch", i)

        training.train_fusion(buddy, fusion_model, e2e_trainset_loader,
                              optim_name="fusion",
                              one_loss=not args.many_loss,
                              init_state_noise=args.init_state_noise,)

    buddy.save_checkpoint("phase_4_fusion")
