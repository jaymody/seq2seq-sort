"""Evaluate model."""
import argparse

from data import CharSortedDataset
from train import evaluate, load_model
from utils import set_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dirpath", type=str)
    parser.add_argument("--model_ckpt", type=str, default="model.ckpt")
    parser.add_argument("--N", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    model = load_model(args.dirpath, args.model_ckpt)
    set_seed(model.hparams.seed)
    pairs = CharSortedDataset(
        args.N, model.hparams.min_length, model.hparams.max_length
    )
    evaluate(model, pairs, args.batch_size)
