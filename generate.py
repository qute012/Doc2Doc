import argparse
import torch

from transformers import BertConfig, EncoderDecoderConfig, BertTokenizer
from transformers import EncoderDecoderModel

from data import get_dataloader

@torch.no_grad()
def generate(model_path):
    device = 'cuda'

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    encoder_config = BertConfig.from_pretrained('bert-base-uncased')
    decoder_config = BertConfig.from_pretrained('bert-base-uncased')
    config = EncoderDecoderConfig.from_encoder_decoder_configs(
        encoder_config, decoder_config
    )
    model = EncoderDecoderModel(config)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model = model.half().eval().to(device)

    data_loader = get_dataloader(
        './msmarco/top1000.dev',
        tokenizer=tokenizer,
        src_max_len=400,
        gen_max_len=150,
        episode=30,
        batch_size=1,
        num_workers=2
    )

    sample_idx = 1
    for batch in data_loader:
        src, tgt = batch

        generated = model.generate(
            input_ids=src["input_ids"].to(device),
            attention_mask=src["attention_mask"].to(device),
            use_cache=True,
            bos_token_id=tokenizer.cls_token_id,
            eos_token_id=tokenizer.sep_token_id,
            pad_token_id=tokenizer.pad_token_id,
            bad_words_ids=[[tokenizer.unk_token_id]],
            num_beams=30,
            do_sample=False,
            temperature=0.7,
            no_repeat_ngram_size=4,
            length_penalty=1.5,
            max_length=150,
        )

        golds = tokenizer.batch_decode(src['input_ids'], skip_special_tokens=True)
        preds = tokenizer.batch_decode(generated, skip_special_tokens=True)

        for gstr, pstr in zip(golds, preds):
            print(f'----------------sample {sample_idx}----------------')
            print(gstr)
            print('====================================================')
            print(pstr)
            print('----------------------------------------------------\n')
            sample_idx += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Relevant Document')
    parser.add_argument('--path', default='', type=str, help="saved model path (e.g. file of .cpkt extention)")
    args = parser.parse_args()
    generate(args.path)

