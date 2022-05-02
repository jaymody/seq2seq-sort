"""Make a prediction."""
import argparse

from train import load_model
from utils import disable_tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dirpath", type=str)
    parser.add_argument("sequences", type=str, nargs="+")
    parser.add_argument("--model_ckpt", type=str, default="model.ckpt")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    disable_tqdm()
    model = load_model(args.dirpath, args.model_ckpt)
    prd_sequences, _, _ = model.predict(args.sequences, batch_size=args.batch_size)
    print("\n".join(prd_sequences))
