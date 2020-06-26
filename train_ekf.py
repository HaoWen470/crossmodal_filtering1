import torch
import fannypack

from lib import panda_datasets, omnipush_datasets
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
        default="ekf",
    )
    parser.add_argument("--data_size", type=int, default=1000, choices=[10, 100, 1000])
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--pretrain", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=15,)
    parser.add_argument("--train", type=str, choices=[ "all", "ekf"], default="all")
    parser.add_argument("--obs_only", action="store_true")
    parser.add_argument("--blackout", type=float, default=0.0)
    parser.add_argument("--mass", action="store_true")
    parser.add_argument("--omnipush", action="store_true")
    parser.add_argument("--hidden_units", type=int, default=128)
    parser.add_argument("--init_state_noise", type=float, default=0.2)
    parser.add_argument("--sequential_image", type=int, default=1)
    parser.add_argument("--start_timestep", type=int, default=0)
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--set_r", type=float, default=None)
    parser.add_argument("--no_proprio", action="store_true")
    parser.add_argument("--measurement_nll", action="store_true")
    parser.add_argument("--ekf_loss", choices=['mse', 'nll', 'mixed'], default="mse")
    parser.add_argument("--learnable_Q", action="store_true")
    parser.add_argument("--learnable_Q_dyn", action="store_true")

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
        'init state noise': args.init_state_noise,
        'sequential_image_rate': args.sequential_image,
        'start_timestep': args.start_timestep,
        'set_r': args.set_r,
        'measurement_nll': args.measurement_nll,
        'ekf_loss': args.ekf_loss,
        'learnable_Q': args.learnable_Q,
        "obs_only": args.obs_only,
        'learnable_Q_dynamics': args.learnable_Q_dyn
    }



    print("Creating model...")
    measurement = PandaEKFMeasurementModel(units=args.hidden_units, use_states= not args.obs_only)
    dynamics = PandaDynamicsModel(use_particles=False, learnable_Q=args.learnable_Q or args.learnable_Q_dyn)
    ekf = KalmanFilterNetwork(dynamics, measurement, R=args.set_r)

    buddy = fannypack.utils.Buddy(experiment_name,
                                  ekf,
                                  optimizer_names=["ekf", "dynamics", "measurement"],
                                  )
    buddy.add_metadata(dataset_args)

    if args.load_checkpoint is not None:
        buddy.load_checkpoint(path = args.load_checkpoint)

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

    if args.train == "all":
        dataloader_dynamics = torch.utils.data.DataLoader(
            dataset_dynamics, batch_size=args.batch, shuffle=True, num_workers=2, drop_last=True)

        for i in range(args.pretrain):
            print("Training dynamics epoch", i)
            training.train_dynamics(buddy, ekf, dataloader_dynamics, optim_name="dynamics")

        #load dynamics data
        dataloader_dynamics = torch.utils.data.DataLoader(
            dynamics_recurrent_trainset, batch_size=args.batch, shuffle=True, num_workers=2, drop_last=True)

        # TRAIN DYNAMICS MODEL
        for i in range(args.pretrain):
            print("Training recurrent dynamics epoch", i)
            training.train_dynamics_recurrent(buddy, ekf, dataloader_dynamics, optim_name="dynamics")

        buddy.save_checkpoint("phase_0_dynamics_pretrain")

        #load measurement data
        measurement_trainset_loader = torch.utils.data.DataLoader(
            dataset_measurement,
            batch_size=args.batch*2,
            shuffle=True,
            num_workers=8)

        # TRAIN MEASUREMENT MODEL
        for i in range(int(args.pretrain/2)):
            print("Training measurement epoch", i)
            training.train_measurement(buddy, ekf, measurement_trainset_loader,
                                       log_interval=20, optim_name="measurement", nll=args.measurement_nll)
            print()

        buddy.save_checkpoint("phase_2_measurement_pretrain")

    #load e2e data
    e2e_trainset_loader = torch.utils.data.DataLoader(e2e_trainset, batch_size=args.batch, shuffle=True, num_workers=2)

    #turn off dynamics Q

    if not args.learnable_Q:
        ekf.dynamics_model.Q.requires_grad = False

    #TRAIN E2D EKF
    for i in range(args.epochs):
        print("Training ekf epoch", i)
        training.train_e2e(buddy, ekf, e2e_trainset_loader,
                           optim_name="ekf",
                           init_state_noise=args.init_state_noise,
                           loss_type=args.ekf_loss)

    buddy.save_checkpoint("phase_3_e2e")
    buddy.save_checkpoint()
