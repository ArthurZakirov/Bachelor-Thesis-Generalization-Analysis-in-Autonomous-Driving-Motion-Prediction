import argparse
from trainer import trainer_ae


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=int, default=0.0001)
    parser.add_argument("--max_epochs", type=int, default=600)

    parser.add_argument("--past_len", type=int, default=20, help="length of past (in timesteps)")
    parser.add_argument("--future_len", type=int, default=60, help="length of future (in timesteps)")
    parser.add_argument("--dim_embedding_key", type=int, default=48)
    parser.add_argument("--dim_clip", type=int, default=180, help="total size of the map window around the vehicle (in pixels)")

    parser.add_argument("--dataset_file_train", type=str, default="Kitti/kitti_train_title.json", help="dataset file with training samples")
    parser.add_argument("--dataset_file_eval", type=str, default="Kitti/kitti_val_title.json", help="dataset file with evaluation samples")
    parser.add_argument("--info", type=str, default='', help='Name of training. '
                                                             'It will be used in tensorboard log and test folder')
    return parser.parse_args()


def main(config):
    print('INIT AUTOENCODER TRAINER')
    t_ae = trainer_ae.Trainer(config)
    print('START TRAINING AUTOENCODER')
    t_ae.fit()



if __name__ == "__main__":
    #python train_ae.py --max_epochs 1 --dataset_file_train openDD/openDD_test_dataset_title.json --dataset_file_eval openDD/openDD_test_dataset_title.json --dim_clip 12
    config = parse_config()
    main(config)
