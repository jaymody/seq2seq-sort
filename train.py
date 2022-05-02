import argparse
import os
import pickle

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import CharSortedDataset, SimpleDataset
from models import Collater, Seq2Seq, pairs_to_tensors
from tokenizers import CharTokenizer
from utils import get_device, score, set_seed

device = get_device()


def train(
    dirpath,
    train_pairs,
    val_pairs,
    test_pairs=None,
    batch_size=512,
    num_workers=8,
    seed=1234,
    args=None,
):
    args = {} if args is None else args
    set_seed(seed)

    # TODO: is it reasonable that the vocab is only built on train pairs?
    src_tokenizer = CharTokenizer()
    trg_tokenizer = CharTokenizer()
    for src, trg in tqdm(train_pairs):
        src_tokenizer.add_sequence(src)
        trg_tokenizer.add_sequence(trg)

    train_tensors = pairs_to_tensors(train_pairs, src_tokenizer, trg_tokenizer)
    val_tensors = pairs_to_tensors(val_pairs, src_tokenizer, trg_tokenizer)

    collate_fn = Collater(src_tokenizer, trg_tokenizer)
    train_dataloader = DataLoader(
        SimpleDataset(train_tensors),
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    val_dataloader = DataLoader(
        SimpleDataset(val_tensors),
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    save_to_pickle = {
        "src_tokenizer.pickle": src_tokenizer,
        "trg_tokenizer.pickle": trg_tokenizer,
    }
    for k, v in save_to_pickle.items():
        with open(os.path.join(dirpath, k), "wb") as fo:
            pickle.dump(v, fo)

    model = Seq2Seq(src_tokenizer, trg_tokenizer, **vars(args)).to(device)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=dirpath,
        filename="model",
        save_top_k=1,
        mode="min",
    )

    trainer = pl.Trainer.from_argparse_args(
        args,
        default_root_dir=dirpath,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, train_dataloader, val_dataloader)  # pylint: disable=no-member

    # not sure why, but after trainer.fit, the model is sent to cpu, so we'll
    # need to send it back to device so evaluate doesn't break
    model.to(device)

    if test_pairs:
        final_score = evaluate(model, test_pairs, batch_size=batch_size)
        with open(os.path.join(dirpath, "eval.txt"), "w") as fo:
            fo.write(f"{final_score:.4f}\n")

    return model


def evaluate(model, test_pairs, batch_size=128):
    src_sequences, trg_sequences = zip(*test_pairs)

    prd_sequences, _, _ = model.predict(src_sequences, batch_size=batch_size)
    assert len(prd_sequences) == len(src_sequences) == len(trg_sequences)

    total_score = 0
    for i, (src, trg, prd) in enumerate(
        tqdm(
            zip(src_sequences, trg_sequences, prd_sequences),
            desc="scoring",
            total=len(src_sequences),
        )
    ):
        pred_score = score(trg, prd)
        total_score += pred_score
        if i < 10:
            print(f"\n\n\n---- Example {i} ----")
            print(f"src = {src}")
            print(f"trg = {trg}")
            print(f"prd = {prd}")
            print(f"score = {pred_score}")

    final_score = total_score / len(prd_sequences)
    print(f"{total_score}/{len(prd_sequences)} = {final_score:.4f}")
    return final_score


def load_model(dirpath, model_ckpt="model.ckpt"):
    with open(os.path.join(dirpath, "src_tokenizer.pickle"), "rb") as fi:
        src_tokenizer = pickle.load(fi)
    with open(os.path.join(dirpath, "trg_tokenizer.pickle"), "rb") as fi:
        trg_tokenizer = pickle.load(fi)
    model = Seq2Seq.load_from_checkpoint(
        os.path.join(dirpath, model_ckpt),
        src_tokenizer=src_tokenizer,
        trg_tokenizer=trg_tokenizer,
    ).to(device)
    return model


def main():
    parser = argparse.ArgumentParser("Train the model.")
    parser.add_argument("dirpath", type=str, default="models/best")
    parser.add_argument("--train_N", type=int, default=100000)
    parser.add_argument("--val_N", type=int, default=20000)
    parser.add_argument("--test_N", type=int, default=20000)
    parser.add_argument("--min_length", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=26)
    parser.add_argument("--train_val_split_ratio", type=float, default=0.95)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser = Seq2Seq.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    os.makedirs(args.dirpath, exist_ok=False)
    train_set_pairs = CharSortedDataset(args.train_N, args.min_length, args.max_length)
    val_set_pairs = CharSortedDataset(args.val_N, args.min_length, args.max_length)
    test_set_pairs = CharSortedDataset(args.test_N, args.min_length, args.max_length)
    train(
        args.dirpath,
        train_set_pairs,
        val_set_pairs,
        test_pairs=test_set_pairs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        args=args,
    )


if __name__ == "__main__":
    main()
