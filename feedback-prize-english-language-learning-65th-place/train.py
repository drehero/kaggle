import argparse
import gc
import os
import pathlib
import pickle
import sys

import numpy as np
import pandas as pd

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from tqdm.auto import tqdm
import wandb

import spacy
import readability
from spellchecker import SpellChecker

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint

from transformers import AutoTokenizer, AdamW
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from data import FBData, collate_fn, AlignedTokenizer, MultiScaleData, get_essay_level_features
from knowledge import n_knowledge_dims
from loss import R2Loss, RDropMSE
from model import Net, MultiScaleModel
from utils import seed_everything, AverageMeter, get_score, str2bool


def train_fn(
        fold,
        train_loader,
        model,
        criterion,
        optimizer,
        epoch,
        scheduler,
        device,
        apex,
        gradient_accumulation_steps,
        max_grad_norm,
        batch_scheduler
    ):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=apex)
    losses = AverageMeter()
    global_step = 0
    with tqdm(train_loader, desc=f"Training Epoch [{epoch+1}]") as pbar:
        for step, (inputs, targets) in enumerate(pbar):
            inputs = collate_fn(inputs)
            for k, v in inputs.items():
                if type(v) is list:
                    for i in range(len(v)):
                        inputs[k][i] = v[i].to(device)
                else:
                    inputs[k] = v.to(device)
            targets = targets.to(device)
            batch_size = targets.shape[0]
            with torch.cuda.amp.autocast(enabled=apex):
                y_pred = model(inputs)
                if args.loss == "rdrop":
                    y_pred_2 = model(inputs)
                    loss = criterion(y_pred, y_pred_2, targets)
                elif args.loss == "r2":
                    loss = criterion(y_pred, targets, epoch+1)
                else:
                    loss = criterion(y_pred, targets)
            if gradient_accumulation_steps > 1:
                loss /= gradient_accumulation_steps
            losses.update(loss.item(), batch_size)
            scaler.scale(loss).backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_grad_norm
            )
            if (step+1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if batch_scheduler:
                    scheduler.step()
            pbar.set_postfix(
                loss=f"{losses.val:.4f} ({losses.avg:.4f})",
                grad=f"{grad_norm:.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.8f}"
            )
            if args.wandb:
                wandb.log({
                    f"[fold{fold}] loss": losses.val,
                    f"[fold{fold}] lr": scheduler.get_lr()[0]
                })
    return losses.avg


def val_fn(val_loader, model, criterion, epoch, device, gradient_accumulation_steps):
    losses = AverageMeter()
    model.eval()
    preds = []
    with tqdm(val_loader, desc=f"Validation Epoch [{epoch+1}]") as pbar:
        for step, (inputs, targets) in enumerate(pbar):
            inputs = collate_fn(inputs)
            for k, v in inputs.items():
                if type(v) is list:
                    for i in range(len(v)):
                        inputs[k][i] = v[i].to(device)
                else:
                    inputs[k] = v.to(device)
            targets = targets.to(device)
            batch_size = targets.shape[0]
            with torch.no_grad():
                y_pred = model(inputs)
                if args.loss == "r2":
                    loss = criterion(y_pred, targets, epoch+1)
                elif args.loss == "rdrop":
                    loss = criterion(y_pred, y_pred, targets)
                else:
                    loss = criterion(y_pred, targets)
            if gradient_accumulation_steps > 1:
                loss /= gradient_accumulation_steps
            losses.update(loss.item(), batch_size)
            preds += [y_pred.to("cpu").numpy()]
            pbar.set_postfix(loss=f"{losses.val:.4f} ({losses.avg:.4f})")
    predictions = np.concatenate(preds)
    return losses.avg, predictions


def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0):
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight", "ln"]
    optimizer_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "lr": encoder_lr,
            "weight_decay": weight_decay
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "lr": encoder_lr,
            "weight_decay": 0
        },
        {
            "params": [p for n, p in model.named_parameters() if "model" not in n and "fc" not in n and "mlp" not in n and "pool" not in n],
            "lr": decoder_lr,
            "weight_decay": 0
        }
    ]
    return optimizer_parameters



def get_optimizer_grouped_parameters(
    model,
    #model_type,
    learning_rate,
    weight_decay
):
    # https://www.kaggle.com/code/rhtsingh/guide-to-huggingface-schedulers-differential-lrs/notebook
    # TODO make compatible for models with different number of hidden layers
    no_decay = ["bias", "LayerNorm.weight"]
    group1 = ["layer.0.", "layer.1.", "layer.2.", "layer.3."]
    group2 = ["layer.4.", "layer.5.", "layer.6.", "layer.7."]    
    group3 = ["layer.8.", "layer.9.", "layer.10.", "layer.11."]
    group_all = [f"layer.{i}." for i in range(model.config.num_hidden_layers)]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all) and "embeddings" in n],
            "weight_decay": weight_decay
        },
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group1)],
            "weight_decay": weight_decay,
            "lr": learning_rate/2.6
        },
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group2)],
            "weight_decay": weight_decay,
            "lr": learning_rate
        },
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group3)],
            "weight_decay": weight_decay,
            "lr": learning_rate*2.6
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],
            "weight_decay": 0.0
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group1)],
            "weight_decay": 0.0,
            "lr": learning_rate/2.6
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group2)],
            "weight_decay": 0.0,
            "lr": learning_rate
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group3)],
            "weight_decay": 0.0,
            "lr": learning_rate*2.6
        },
        {
            "params": [p for n, p in model.named_parameters() if not "encoder" in n and not "embeddings" in n and not any(nd in n for nd in no_decay)],
            "lr": learning_rate*20,
            "weight_decay": 0.0
        },
    ]
    return optimizer_grouped_parameters

def get_scheduler(optimizer, num_warmup_steps, num_training_steps, scheduler):
    if scheduler == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    elif scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    return scheduler


def pretrain_loop(external_df, train_df, args):
    print(f"========== Pre training ==========")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # This won't work for essay level feats
    train_data = FBData(external_df, args, None)
    val_data = FBData(train_df, args, None)
    val_targets = train_df[args.target_names].values
    
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        #num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader( 
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        #num_workers=args.workers,
        pin_memory=True,
        drop_last=False
    )

    # This wont work for multiscale
    model = Net(args, config_path=None, pretrained=True)
    torch.save(model.config, args.out_path / "pretrained-config.pt")
    model.to(device)

    # This wont work for llrd
    optimizer_parameters = get_optimizer_params(
        model,
        encoder_lr=args.encoder_lr,
        decoder_lr=args.decoder_lr,
        weight_decay=args.weight_decay
    )
    optimizer = AdamW(
        optimizer_parameters,
        lr=args.encoder_lr,
        eps=args.eps,
        betas=(args.b1, args.b2),
        correct_bias=not args.bertadam
    )

    num_training_steps = args.epochs * len(train_loader)
    scheduler = get_scheduler(
        optimizer,
        args.warmup_steps,
        num_training_steps,
        args.scheduler
    )

    if args.loss == "l1":
        criterion = nn.SmoothL1Loss(reduction="mean")
    elif args.loss == "rmse":
        criterion = RMSELoss(reduction="mean")
    elif args.loss == "r2":
        criterion = R2Loss(args.epochs, reduction="mean")
    elif args.loss == "rdrop":
        criterion = RDropMSE(alpha=args.alpha, reduction="mean")

    scaler = torch.cuda.amp.GradScaler(enabled=args.apex)
    losses = AverageMeter()
    global_step = 0
    best_score = np.inf
    for epoch in range(args.epochs):
        with tqdm(train_loader, desc=f"Pre Training Epoch [{epoch+1}]") as pbar:
            for step, (inputs, targets) in enumerate(pbar):
                global_step += 1
                model.train()
                inputs = collate_fn(inputs)
                for k, v in inputs.items():
                    if type(v) is list:
                        for i in range(len(v)):
                            inputs[k][i] = v[i].to(device)
                    else:
                        inputs[k] = v.to(device)
                targets = targets.to(device)
                batch_size = targets.shape[0]
                with torch.cuda.amp.autocast(enabled=args.apex):
                    y_pred = model(inputs)
                    if args.loss == "rdrop":
                        y_pred_2 = model(inputs)
                        loss = criterion(y_pred, y_pred_2, targets)
                    elif args.loss == "r2":
                        loss = criterion(y_pred, targets, epoch+1)
                    else:
                        loss = criterion(y_pred, targets)
                if args.gradient_accumulation_steps > 1:
                    loss /= args.gradient_accumulation_steps
                losses.update(loss.item(), batch_size)
                scaler.scale(loss).backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm
                )
                if (step+1) % args.gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    if args.batch_scheduler:
                        scheduler.step()
                if (global_step) % args.pretrain_eval_steps == 0:
                    # eval
                    avg_val_loss, predictions = val_fn(
                        val_loader,
                        model,
                        criterion,
                        epoch,
                        device,
                        args.gradient_accumulation_steps
                    )
                    score, scores = get_score(val_targets, predictions, args.target_scaler)

                    if best_score > score:
                        best_score = score
                        print(f"Epoch {epoch+1} - Save Best Score: {best_score:.4f}")
                        torch.save({
                            "model": model.state_dict(),
                            "predictions": predictions
                        }, args.out_path / f"{args.model.replace('/', '-')}-pretrained.pt")
        
                    if args.wandb:
                        wandb.log({
                            f"[pretraining] epoch": epoch+1, 
                            f"[pretraining] avg_val_loss": avg_val_loss,
                            f"[pretraining] score": score
                        })

                if args.wandb:
                    wandb.log({
                        f"[pretraining] loss": losses.val,
                        f"[pretraining] lr": scheduler.get_lr()[0]
                    })
                    
                pbar.set_postfix(
                    loss=f"{losses.val:.4f} ({losses.avg:.4f})",
                    grad=f"{grad_norm:.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.8f}"
                )


def train_loop(df, fold, args):
    print(f"========== fold: {fold} training ==========")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_folds = df[df["fold"] != fold].reset_index(drop=True)
    val_folds = df[df["fold"] == fold].reset_index(drop=True)
    val_targets = val_folds[args.target_names].values

    if args.essay_level_feats:
        essay_level_feats = torch.stack(train_folds["essay_level_feats"].to_list())
        essay_feats_scaler = StandardScaler().fit(essay_level_feats)
        with open(args.out_path / f"essay_feats_scaler_f{fold}.pickle", "wb") as f:
            pickle.dump(essay_feats_scaler, f)
    else:
        essay_feats_scaler = None

    if args.multi_scale:
        train_data = MultiScaleData(train_folds, args)
        val_data = MultiScaleData(val_folds, args)
    else:
        train_data = FBData(train_folds, args, essay_feats_scaler)
        val_data = FBData(val_folds, args, essay_feats_scaler)

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        #num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader( 
        val_data,
        batch_size=args.batch_size*2,
        shuffle=False,
        #num_workers=args.workers,
        pin_memory=True,
        drop_last=False
    )

    if args.pretrain:
        config_path = args.out_path / "pretrained-config.pt"
    else:
        config_path = None
        
    if args.multi_scale:
        model = MultiScaleModel(args, config_path=config_path, pretrained=not args.pretrain) 
    else:
        model = Net(args, config_path=config_path, pretrained=not args.pretrain)
	
    if args.pretrain:
        state = torch.load(
            args.out_path / f"{args.model.replace('/', '-')}-pretrained.pt",
            map_location=torch.device("cpu")
        )
        model.load_state_dict(state["model"])

    if args.multi_scale:
        torch.save(model.word_doc_model.config, args.out_path / "word_doc_model_config.pt")
        torch.save(model.seg_model.config, args.out_path / "seg_model_config.pt")
    else:
        torch.save(model.config, args.out_path / "config.pt")
    model.to(device)

    if args.llrd:
        optimizer_parameters = get_optimizer_grouped_parameters(
            model,
            #model_type=args.model_type,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        optimizer = AdamW(
            optimizer_parameters,
            lr=args.learning_rate,
            eps=args.eps,
            correct_bias=not args.bertadam
        )
    else:
        optimizer_parameters = get_optimizer_params(
            model,
            encoder_lr=args.encoder_lr,
            decoder_lr=args.decoder_lr,
            weight_decay=args.weight_decay
        )
        optimizer = AdamW(
            optimizer_parameters,
            lr=args.encoder_lr,
            eps=args.eps,
            betas=(args.b1, args.b2),
            correct_bias=not args.bertadam
        )

    num_training_steps = len(train_folds) // args.batch_size * args.epochs
    scheduler = get_scheduler(
        optimizer,
        args.warmup_steps,
        num_training_steps,
        args.scheduler
    )

    if args.loss == "l1":
        criterion = nn.SmoothL1Loss(reduction="mean")
    elif args.loss == "rmse":
        criterion = RMSELoss(reduction="mean")
    elif args.loss == "r2":
        criterion = R2Loss(args.epochs, reduction="mean")
    elif args.loss == "rdrop":
        criterion = RDropMSE(alpha=args.alpha, reduction="mean")

    best_score = np.inf
    for epoch in range(args.epochs):
        # train
        avg_loss = train_fn(
            fold,
            train_loader,
            model,
            criterion,
            optimizer,
            epoch,
            scheduler,
            device,
            apex=args.apex,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
            batch_scheduler=args.batch_scheduler
        )

        # eval
        avg_val_loss, predictions = val_fn(
            val_loader,
            model,
            criterion,
            epoch,
            device,
            args.gradient_accumulation_steps
        )
        score, scores = get_score(val_targets, predictions, args.target_scaler)

        if args.wandb:
            wandb.log({
                f"[fold{fold}] epoch": epoch+1, 
                f"[fold{fold}] avg_train_loss": avg_loss, 
                f"[fold{fold}] avg_val_loss": avg_val_loss,
                f"[fold{fold}] score": score
            })

        if best_score > score:
            best_score = score
            print(f"Epoch {epoch+1} - Save Best Score: {best_score:.4f}")
            torch.save({
                "model": model.state_dict(),
                "predictions": predictions
            }, args.out_path / f"{args.model.replace('/', '-')}-fold{fold}.pt")

    predictions = torch.load(
        args.out_path / f"{args.model.replace('/', '-')}-fold{fold}.pt",
        map_location=torch.device("cpu")
    )["predictions"] 
    val_folds[[f"pred_{c}" for c in args.target_names]] = predictions

    torch.cuda.empty_cache()
    gc.collect()

    return val_folds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", type=str2bool, default=False)
    parser.add_argument("--wandb", type=str2bool, required=True)
    parser.add_argument("--in-path", type=pathlib.Path, required=True)
    parser.add_argument("--out-path", type=pathlib.Path, default=pathlib.Path(__file__).absolute().parent)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=str2bool, default=True)
    # data
    parser.add_argument("--scale-targets", type=str2bool, required=True)
    # features
    parser.add_argument("--token-level-feats", type=str2bool, required=True)
    parser.add_argument("--essay-level-feats", type=str2bool, required=True)
    # model
    parser.add_argument("--model", type=str, required=True)
    #parser.add_argument("--model-type", type=str, required=True)
    parser.add_argument("--multi-scale", type=str2bool, required=True)
    parser.add_argument("--pooling", type=str, required=True)
    parser.add_argument("--reinit-layers", type=int, required=True)
    parser.add_argument("--hidden-dropout", type=float, default=0)
    parser.add_argument("--hidden-dropout-prob", type=float, default=0)
    parser.add_argument("--attention-dropout", type=float, default=0)
    parser.add_argument("--attention-probs-dropout-prob", type=float, default=0)
    # training
    parser.add_argument("--pretrain", type=str2bool, required=True)
    parser.add_argument("--pretrain_eval_steps", type=int, default=300)
    parser.add_argument("--loss", type=str, required=True, choices=["l1", "rmse", "r2", "rdrop"])
    parser.add_argument("--alpha", type=float, default=5)
    parser.add_argument("--folds", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--apex", type=str2bool, default=True)
    parser.add_argument("--bertadam", type=str2bool, required=True)
    parser.add_argument("--gradient_checkpointing", type=str2bool, default=True)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1000)
    parser.add_argument("--batch-scheduler", type=str2bool, default=True)
    parser.add_argument("--llrd", type=str2bool, required=True)
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--decoder-lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--b1", type=float, default=0.9)
    parser.add_argument("--b2", type=float, default=0.999)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "linear"])
    parser.add_argument("--warmup-steps", type=int, default=0)

    args = parser.parse_args()

    seed_everything(args.seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "true" 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    train = pd.read_csv(args.in_path / "feedback-prize-english-language-learning/train.csv")
    if args.pretrain:
        external = pd.read_csv(args.in_path / "fb3-pl/fb3-external-pl.csv")
    
    if args.debug:
        args.epochs = 3
        args.batch_size = 2
        args.folds = 2
        args.wandb = False
        args.pretrain_eval_steps = 1 

    args.target_names = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]

    #scale targets
    if args.scale_targets:
        target_scaler = MinMaxScaler().fit(train[args.target_names].values)
        args.target_scaler = target_scaler
        with open(args.out_path / "target-scaler.pickle", "wb") as f:
            pickle.dump(target_scaler, f)
    else:
        args.target_scaler = None

    # token level feats
    if args.token_level_feats:
        hf_tokenizer = AutoTokenizer.from_pretrained(args.model)
        tokenizer = AlignedTokenizer(hf_tokenizer)
        args.pooling = "none"
        args.n_knowledge_dims = n_knowledge_dims
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        tokenizer.save_pretrained(args.out_path / "tokenizer/")
    args.tokenizer = tokenizer

    # essay level features
    if args.essay_level_feats:
        nlp = spacy.load("en_core_web_sm")
        spell = SpellChecker()
        train["essay_level_feats"] = train["full_text"].apply(lambda x: get_essay_level_features(x, nlp, spell))
        args.essay_level_feats_dim = len(train.loc[0, "essay_level_feats"])

    # get max len
    l = []
    for text in train["full_text"].fillna("").values:
        if args.token_level_feats:
            l += [len(args.tokenizer.tokenizer(text, add_special_tokens=False)["input_ids"])]
        else:
            l += [len(args.tokenizer(text, add_special_tokens=False)["input_ids"])]
    args.max_len = max(l)

    # multi scale
    if args.multi_scale:
        args.chunk_sizes = [90, 30, 130, 10]

    # llrd
    if args.llrd:
        args.learning_rate = 5e-5
        args.weight_decay = 0.01 
    
    # save training args
    with open(args.out_path / "train-args.pickle", "wb") as f:
        pickle.dump(args, f)

    # wandb
    if args.wandb:
        wandb.init(
            project="FB3", 
            name=args.model,
            config=vars(args),
            group=args.model,
            job_type="train",
        )

    # pretrain
    if args.pretrain and args.debug:
        pretrain_loop(external.iloc[:5, :], train.iloc[:5, :], args)
    elif args.pretrain:
        pretrain_loop(external, train, args)

    # create folds
    train["fold"] = -1
    mskf = MultilabelStratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for f, (train_idx, val_idx) in enumerate(mskf.split(train, train[args.target_names])):
        train.loc[val_idx, "fold"] = f

    if args.debug:
        train = train.groupby("fold").head(2).reset_index(drop=True)

    oof_df = pd.DataFrame()
    for fold in range(args.folds):
        _oof_df = train_loop(train, fold, args)
        oof_df = pd.concat([oof_df, _oof_df])
    oof_df = oof_df.reset_index(drop=True)

    pred_cols = [f"pred_{c}" for c in args.target_names]
    if args.scale_targets:
        # unscale predictions
        oof_df[pred_cols] = target_scaler.inverse_transform(oof_df[pred_cols])

    oof_df.to_pickle(args.out_path / "oof_df.pickle")

    targets = oof_df[args.target_names].values
    pred = oof_df[pred_cols].values
    score, scores = get_score(targets, pred, None)

    if args.wandb:
        wandb.finish()

    print(score)
    print(scores)
