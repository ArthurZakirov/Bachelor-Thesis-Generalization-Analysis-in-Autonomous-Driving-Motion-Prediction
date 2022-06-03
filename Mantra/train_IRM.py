import argparse
from trainer import trainer_IRM


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



    # MODEL CONTROLLER
    parser.add_argument("--model", type=str, default='pretrained_models/model_controller/model_controller')

    parser.add_argument("--saved_memory", default=True)
    parser.add_argument("--saveImages", default=True, help="plot qualitative examples in tensorboard")
    parser.add_argument("--dataset_file_train", type=str, default="Kitti/kitti_train_title.json",
                        help="dataset file with training samples")
    parser.add_argument("--dataset_file_eval", type=str, default="Kitti/kitti_val_title.json",
                        help="dataset file with evaluation samples")
    parser.add_argument("--info", type=str, default='', help='Name of training. '
                                                             'It will be used in tensorboard log and test folder')
    return parser.parse_args()


def main(config):
    t = trainer_IRM.Trainer(config)
    print('start training IRM')
    t.fit()


if __name__ == "__main__":
    #python train_IRM.py --max_epochs 1 --model "training/training_controller/2021-08-17 13:38:56_/model_controller_2021-08-17 13:38:56" --dataset_file_train Kitti/kitti_sample_title.json --dataset_file_eval Kitti/kitti_sample_title.json
    config = parse_config()
    main(config)
