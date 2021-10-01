from transformers import BertTokenizer

from data import get_dataloader
from model.doc2doc.bert2bert import Doc2Doc

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_dataloader = get_dataloader(
        './msmarco/top1000.dev',
        tokenizer=tokenizer,
        src_max_len=400,
        gen_max_len=150,
        episode=100000,
        batch_size=24,
        shuffle=True,
        num_workers=8
    )

    valid_dataloader = get_dataloader(
        './msmarco/top1000.dev',
        tokenizer=tokenizer,
        src_max_len=400,
        gen_max_len=150,
        episode=1000,
        batch_size=24,
        num_workers=8
    )

    model = Doc2Doc(
        pretrained_model_name_or_path='bert-base-uncased',
        pad_id=tokenizer.pad_token_id,
        bos_id=tokenizer.cls_token_id,
        eos_id=tokenizer.sep_token_id
    )

    logger = TensorBoardLogger("d2d_tb_log", name="doc2doc")

    """
        trainer = Trainer(
            gpus=0
        )
        """
    trainer = Trainer(
        max_epochs=100,
        max_steps=1000000,
        # progress_bar_refresh_rate=0,
        gpus=4,
        distributed_backend='ddp',
        precision=16,
        # amp_backend="apex",
        # amp_level='O2',
        logger=logger,
        callbacks=[EarlyStopping(monitor='val_loss')]
    )
    trainer.fit(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloaders=valid_dataloader
    )
