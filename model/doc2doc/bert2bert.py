import torch
import torch.nn as nn

from transformers import BertConfig, EncoderDecoderConfig
from transformers import BertModel, EncoderDecoderModel
from transformers import AdamW
from transformers.models.bart.modeling_bart import shift_tokens_right
import pytorch_lightning as pl

from .scheduler import LinearSchedulerWithWarmup


class Doc2Doc(pl.LightningModule):
    def __init__(
            self,
            pretrained_model_name_or_path='bert-base-uncased',
            pad_id=0,
            bos_id=101,
            eos_id=102,
            checkpoint_interval=2000
    ):
        super(Doc2Doc, self).__init__()

        encoder_config = BertConfig.from_pretrained(pretrained_model_name_or_path)
        decoder_config = BertConfig.from_pretrained(pretrained_model_name_or_path)
        config = EncoderDecoderConfig.from_encoder_decoder_configs(
            encoder_config, decoder_config
        )

        self.model = EncoderDecoderModel(config)

        state_dict = BertModel.from_pretrained(pretrained_model_name_or_path).state_dict()
        self.model.encoder.load_state_dict(state_dict)
        self.model.decoder.bert.load_state_dict(state_dict, strict=False)
        self.criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=pad_id)

        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id

        self.checkpoint_interval =checkpoint_interval

    def training_step(self, batch, batch_idx):
        src, tgt = batch
        target = tgt.contiguous()
        decoder_input_ids = shift_tokens_right(tgt, self.pad_id, self.bos_id)
        logits = self.model(**src, decoder_input_ids=decoder_input_ids)[0]
        loss = self.criterion(logits.view(-1, logits.size(-1)), target.view(-1))
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.save_model()
        return {'loss': loss, 'pregress_bar': {'training_loss': loss}}

    def validation_step(self, batch, batch_idx):
        src, tgt = batch
        target = tgt.contiguous()
        decoder_input_ids = shift_tokens_right(tgt, self.pad_id, self.bos_id)
        logits = self.model(**src, decoder_input_ids=decoder_input_ids)[0]
        loss = self.criterion(logits.view(-1, logits.size(2)), target.view(-1))
        self.log('val_loss', loss)
        return {'val_loss': loss}

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=5e-3,
            #betas=(0.9, 0.98)
        )

        lr_scheduler = LinearSchedulerWithWarmup(
            optimizer,
            1000000,
            num_warmup_steps=4000
        )
        return [optimizer], [lr_scheduler]

    def save_model(self):
        if self.trainer.global_rank == 0 and self.global_step % self.checkpoint_interval == 0:
            torch.save(
                self.model.state_dict(),
                "checkpoint/" + "{}_steps.pth".format(str(self.global_step))
            )
