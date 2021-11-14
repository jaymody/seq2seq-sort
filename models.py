import argparse

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import SimpleDataset
from layers import Decoder, Encoder
from utils import get_device

device = get_device()


def sequence_to_tensor(sequence, tokenizer):
    indexes = [tokenizer.token2index[w] for w in tokenizer.sequence_to_tokens(sequence)]
    indexes = [tokenizer.SOS_IDX] + indexes + [tokenizer.EOS_IDX]
    return torch.LongTensor(indexes)


def pairs_to_tensors(pairs, src_tokenizer, trg_tokenizer):
    tensors = [
        (sequence_to_tensor(src, src_tokenizer), sequence_to_tensor(trg, trg_tokenizer))
        for src, trg in tqdm(pairs, desc="creating tensors")
    ]
    return tensors


class Collater:
    def __init__(self, src_tokenizer, trg_tokenizer=None, predict=False):
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer
        self.predict = predict

    def __call__(self, batch):
        # TODO: try pack_padded_sequence for faster processing
        if self.predict:
            # batch = src_tensors in predict mode
            return nn.utils.rnn.pad_sequence(
                batch, batch_first=True, padding_value=self.src_tokenizer.PAD_IDX
            )

        src_tensors, trg_tensors = zip(*batch)
        src_tensors = nn.utils.rnn.pad_sequence(
            src_tensors, batch_first=True, padding_value=self.src_tokenizer.PAD_IDX
        )
        trg_tensors = nn.utils.rnn.pad_sequence(
            trg_tensors, batch_first=True, padding_value=self.trg_tokenizer.PAD_IDX
        )
        return src_tensors, trg_tensors


class Seq2Seq(pl.LightningModule):
    def __init__(
        self,
        src_tokenizer,
        trg_tokenizer,
        max_length=32,
        hid_dim=256,
        enc_layers=3,
        dec_layers=3,
        enc_heads=8,
        dec_heads=8,
        enc_pf_dim=512,
        dec_pf_dim=512,
        enc_dropout=0.1,
        dec_dropout=0.1,
        lr=0.0005,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()
        del self.hparams["src_tokenizer"]
        del self.hparams["trg_tokenizer"]

        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer

        self.encoder = Encoder(
            src_tokenizer.n_tokens,
            hid_dim,
            enc_layers,
            enc_heads,
            enc_pf_dim,
            enc_dropout,
            device,
        )

        self.decoder = Decoder(
            trg_tokenizer.n_tokens,
            hid_dim,
            dec_layers,
            dec_heads,
            dec_pf_dim,
            dec_dropout,
            device,
        )

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.trg_tokenizer.PAD_IDX)
        self.initialize_weights()
        self.to(device)

    def initialize_weights(self):
        def _initialize_weights(m):
            if hasattr(m, "weight") and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight.data)

        self.encoder.apply(_initialize_weights)
        self.decoder.apply(_initialize_weights)

    def make_src_mask(self, src):

        # src = [batch size, src len]

        src_mask = (src != self.src_tokenizer.PAD_IDX).unsqueeze(1).unsqueeze(2)

        # src_mask = [batch size, 1, 1, src len]

        return src_mask

    def make_trg_mask(self, trg):

        # trg = [batch size, trg len]

        trg_pad_mask = (trg != self.trg_tokenizer.PAD_IDX).unsqueeze(1).unsqueeze(2)

        # trg_pad_mask = [batch size, 1, 1, trg len]

        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len)).type_as(trg)).bool()

        # trg_sub_mask = [trg len, trg len]

        trg_mask = trg_pad_mask & trg_sub_mask

        # trg_mask = [batch size, 1, trg len, trg len]

        return trg_mask

    def forward(self, src, trg):

        # src = [batch size, src len]
        # trg = [batch size, trg len]

        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        # src_mask = [batch size, 1, 1, src len]
        # trg_mask = [batch size, 1, trg len, trg len]

        enc_src = self.encoder(src, src_mask)

        # enc_src = [batch size, src len, hid dim]

        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)

        # output = [batch size, trg len, output dim]
        # attention = [batch size, n heads, trg len, src len]

        return output, attention

    def predict(self, sequences, batch_size=128):
        """Efficiently predict a list of sequences"""
        pred_tensors = [
            sequence_to_tensor(sequence, self.src_tokenizer)
            for sequence in tqdm(sequences, desc="creating prediction tensors")
        ]

        collate_fn = Collater(self.src_tokenizer, predict=True)
        pred_dataloader = DataLoader(
            SimpleDataset(pred_tensors),
            batch_size=batch_size,
            collate_fn=collate_fn,
        )

        sequences = []
        tokens = []
        attention = []
        for batch in tqdm(pred_dataloader, desc="predict batch num"):
            preds = self.predict_batch(batch.to(device))
            pred_sequences, pred_tokens, pred_attention = preds
            sequences.extend(pred_sequences)
            tokens.extend(pred_tokens)
            attention.extend(pred_attention)

        # sequences = [num pred sequences]
        # tokens = [num pred sequences, trg len]
        # attention = [num pred sequences, n heads, trg len, src len]

        return sequences, tokens, attention

    def predict_batch(self, batch):
        """Predicts on a batch of src_tensors."""
        # batch = src_tensor when predicting = [batch_size, src len]

        src_tensor = batch
        src_mask = self.make_src_mask(batch)

        # src_mask = [batch size, 1, 1, src len]

        enc_src = self.encoder(src_tensor, src_mask)

        # enc_src = [batch size, src len, hid dim]

        trg_indexes = [[self.trg_tokenizer.SOS_IDX] for _ in range(len(batch))]

        # trg_indexes = [batch_size, cur trg len = 1]

        trg_tensor = torch.LongTensor(trg_indexes).to(self.device)

        # trg_tensor = [batch_size, cur trg len = 1]
        # cur trg len increases during the for loop up to the max len

        for _ in range(self.hparams.max_length):

            trg_mask = self.make_trg_mask(trg_tensor)

            # trg_mask = [batch size, 1, cur trg len, cur trg len]

            output, attention = self.decoder(trg_tensor, enc_src, trg_mask, src_mask)

            # output = [batch size, cur trg len, output dim]

            preds = output.argmax(2)[:, -1].reshape(-1, 1)

            # preds = [batch_size, 1]

            trg_tensor = torch.cat((trg_tensor, preds), dim=-1)

            # trg_tensor = [batch_size, cur trg len], cur trg len increased by 1

        src_tensor = src_tensor.detach().cpu().numpy()
        trg_tensor = trg_tensor.detach().cpu().numpy()
        attention = attention.detach().cpu().numpy()

        pred_tokens = []
        pred_sequences = []
        pred_attention = []
        for src_indexes, trg_indexes, attn in zip(src_tensor, trg_tensor, attention):
            # trg_indexes = [trg len = max len (filled with eos if max len not needed)]
            # src_indexes = [src len = len of longest sequence (padded if not longest)]

            # indexes where first eos tokens appear
            src_eosi = np.where(src_indexes == self.src_tokenizer.EOS_IDX)[0][0]
            _trg_eosi_arr = np.where(trg_indexes == self.trg_tokenizer.EOS_IDX)[0]
            if len(_trg_eosi_arr) > 0:  # check that an eos token exists in trg
                trg_eosi = _trg_eosi_arr[0]
            else:
                trg_eosi = len(trg_indexes)

            # cut target indexes up to first eos token and also exclude sos token
            trg_indexes = trg_indexes[1:trg_eosi]

            # attn = [n heads, trg len=max len, src len=max len of sequence in batch]
            # we want to keep n heads, but we'll cut trg len and src len up to
            # their first eos token
            attn = attn[:, :trg_eosi, :src_eosi]  # cut attention for trg eos tokens

            tokens = [self.trg_tokenizer.index2token[index] for index in trg_indexes]
            sequence = self.trg_tokenizer.tokens_to_sequence(tokens)
            pred_tokens.append(tokens)
            pred_sequences.append(sequence)
            pred_attention.append(attn)

        # pred_sequences = [batch_size]
        # pred_tokens = [batch_size, trg len]
        # attention = [batch size, n heads, trg len (varies), src len (varies)]

        return pred_sequences, pred_tokens, pred_attention

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def training_step(self, batch, batch_idx):
        src, trg = batch

        output, _ = self(src, trg[:, :-1])

        # output = [batch size, trg len - 1, output dim]
        # trg = [batch size, trg len]

        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)

        # output = [batch size * trg len - 1, output dim]
        # trg = [batch size * trg len - 1]

        loss = self.criterion(output, trg)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        src, trg = batch

        output, _ = self(src, trg[:, :-1])

        # output = [batch size, trg len - 1, output dim]
        # trg = [batch size, trg len]

        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)

        # output = [batch size * trg len - 1, output dim]
        # trg = [batch size * trg len - 1]

        loss = self.criterion(output, trg)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    @staticmethod
    def add_model_specific_args(parent_parser):
        _parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        _parser.add_argument("--hid_dim", type=int, default=256)
        _parser.add_argument("--enc_layers", type=int, default=3)
        _parser.add_argument("--dec_layers", type=int, default=3)
        _parser.add_argument("--enc_heads", type=int, default=8)
        _parser.add_argument("--dec_heads", type=int, default=8)
        _parser.add_argument("--enc_pf_dim", type=int, default=512)
        _parser.add_argument("--dec_pf_dim", type=int, default=512)
        _parser.add_argument("--enc_dropout", type=float, default=0.1)
        _parser.add_argument("--dec_dropout", type=float, default=0.1)
        _parser.add_argument("--lr", type=float, default=0.0005)
        return _parser
