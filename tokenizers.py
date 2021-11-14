class Tokenizer:
    PAD_TKN = "<pad>"
    SOS_TKN = "<sos>"
    EOS_TKN = "<eos>"
    UNK_TKN = "<unk>"

    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2
    UNK_IDX = 3

    def __init__(self):
        self.token2count = {}
        self.token2index = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.index2token = {v: k for k, v in self.token2index.items()}
        self.n_tokens = 4
        self.max_length = 0

    def sequence_to_tokens(self, sequence):
        raise NotImplementedError()

    def tokens_to_sequence(self, tokens):
        raise NotImplementedError()

    def add_sequence(self, sequence):
        tokens = self.sequence_to_tokens(sequence)

        if len(tokens) > self.max_length:
            self.max_length = len(tokens)

        for token in tokens:
            self.add_token(token)

    def add_token(self, token):
        if token not in self.token2index:
            self.token2index[token] = self.n_tokens
            self.token2count[token] = 1
            self.index2token[self.n_tokens] = token
            self.n_tokens += 1
        else:
            self.token2count[token] += 1


class CharTokenizer(Tokenizer):
    def sequence_to_tokens(self, sequence):
        return list(sequence)

    def tokens_to_sequence(self, tokens):
        return "".join(tokens)


class WordTokenizer(Tokenizer):
    def sequence_to_tokens(self, sequence):
        return sequence.split()

    def tokens_to_sequence(self, tokens):
        return " ".join(tokens)
