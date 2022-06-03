import argparse
from trainer import trainer_controllerMem


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=int, default=0.0001)
    parser.add_argument("--max_epochs", type=int, default=600)

    parser.add_argument("--past_len", type=int, default=20)
    parser.add_argument("--future_len", type=int, default=60)
    parser.add_argument("--preds", type=int, default=5)
    parser.add_argument("--dim_embedding_key", type=int, default=48)
    parser.add_argument("--dim_clip", type=int, default=180,
                        help="total size of the map window around the vehicle (in pixels)")

    parser.add_argument("--model_ae", type=str, default='pretrained_models/model_AE/model_ae')
    parser.add_argument("--dataset_file_train", type=str, default="Kitti/kitti_train_title.json",
                        help="dataset file with training samples")
    parser.add_argument("--dataset_file_eval", type=str, default="Kitti/kitti_val_title.json",
                        help="dataset file with evaluation samples")
    parser.add_argument("--info", type=str, default='', help='Name of training. '
                                                             'It will use in tensorboard log and test folder')
    return parser.parse_args()


def main(config):
    print('Start training writing controller')
    t = trainer_controllerMem.Trainer(config)
    t.fit()


if __name__ == "__main__":
    #python train_controllerMem.py --max_epochs 1 --model_ae "training/training_ae/2021-08-17 13_/model_ae_2021-08-17 13" --dataset_file_train Kitti/kitti_sample_title.json --dataset_file_eval Kitti/kitti_sample_title.json
    config = parse_config()
    main(config)
