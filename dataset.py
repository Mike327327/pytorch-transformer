import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.seq_len = seq_len

        self.ds = ds                                # dataset
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64) # start of sentence token
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64) # end of sentence token
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64) # padding token

    # length of the dataset
    def __len__(self):
        return len(self.ds)

    # get the item at the index idx
    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # Transform the text into tokens and then into ids
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # the model works with fixed size sequences, so we need to make sure that the input and output sequences have the same length
        # here, we calculate how many padding tokens we need to add to the input and output sequences to make them be of length seq_len
        # Add sos, eos and padding to each sentence
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # We will add <s> and </s>, thus -2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 # We will only add <s>

        # Make sure the number of padding tokens is not negative. If it is, the sentence is too long
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # build input tensors for 1) the encoder, 2) the decoder and 3) the label (expected output of the decoder)
        # Add <s> and </s> token
        encoder_input = torch.cat(
            [
                self.sos_token,                                                                     # start of sentence token
                torch.tensor(enc_input_tokens, dtype=torch.int64),                                  # source tokens   
                self.eos_token,                                                                     # end of sentence token                            
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),         # padding tokens
            ],
            dim=0,
        )

        # Add only <s> token
        decoder_input = torch.cat(
            [
                self.sos_token,                                                                    # start of sentence token
                torch.tensor(dec_input_tokens, dtype=torch.int64),                                 # target tokens
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),        # padding tokens
            ],
            dim=0,
        )

        # Add only </s> token (what we expect as output of the decoder)
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Double check the size of the tensors to make sure they are all seq_len long
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len
        
        # If encoder_input = [1, 2, 3, 4, PAD, PAD], then:
        #   encoder_mask = [1, 1, 1, 1, 0, 0]
        # The mask tells the model that the tokens at indices 4 and 5 are padding, so it should ignore them.
        
        # decoder_input = [SOS, 1, 2, 3] -> causal_mask(4)
        # causal_mask(4) will return:
        # [[True, False, False, False],
        #  [True, True, False, False],
        #  [True, True, True, False],
        #  [True, True, True, True]]


        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            # all tokens that are not padding are OK
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len),
            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }
    
def causal_mask(size):
    # triu returns the upper triangular part of a matrix (2-D tensor) with the elements below the k-th (in our case 1st) diagonal zeroed
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    # but we want the elements above the diagonal to be zeroed, so we negate the mask
    # everything that equals 0 will become 1 and everything that equals 1 will become 0 (inverted mask)
    return mask == 0