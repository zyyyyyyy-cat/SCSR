from fseft.modeling import peft
from fseft.modeling import blackbox


def get_adapt_config(args):
    print(f"get_adapt_config args.freqfit {args.freqfit}")
    # Training from scratch/fine-tuning
    if args.method == "scratch" or args.method == "FT": # ori_lr = 1e-4
        adapt_hp = {"batch_size": 1, "num_samples": 6, "epochs": 200, "lr": args.lr, "scheduler": True, "warmup_epoch": 1,
                    "set_peft": peft.FT.set_peft, "set_bb": peft.FT.set_bb,
                    "set_training_mode": peft.FT.set_training_mode, "pat": 20,
                    "decoder": args.decoder, "bottleneck": args.bottleneck}
    # Linear Probe
    elif args.method in ["LP", "generalization"]: # ori_lr = 1e-2
        adapt_hp = {"batch_size": 1, "num_samples": 6, "epochs": 200, "lr": args.lr, "scheduler": True, "warmup_epoch": 1,
                    "set_peft": blackbox.LP.set_peft, "set_bb": blackbox.LP.set_bb,
                    "set_training_mode": blackbox.LP.set_training_mode, "pat": 20,
                    "decoder": args.decoder, "bottleneck": args.bottleneck}
        if args.decoder != "frozen":  # Decrease learning rate if we also update the model decoder
            adapt_hp["lr"] = 1e-4
    # bb3DAdapter / bbAdapter
    elif args.method == "bbAdapter" or args.method == "bb3DAdapter":  # ori_lr = 1e-4
        adapt_hp = {"batch_size": 1, "num_samples": 6, "epochs": 200, "lr": args.lr, "scheduler": True, "warmup_epoch": 1,
                    "set_peft": blackbox.Adapter.set_peft, "set_bb": blackbox.Adapter.set_bb,
                    "set_training_mode": blackbox.Adapter.set_training_mode, "pat": 20,
                    "decoder": args.decoder, "bottleneck": args.bottleneck}
    # bbFreqFit
    elif args.method == "bbFreqFit":  # ori_lr = 1e-4
        adapt_hp = {"batch_size": 1, "num_samples": 6, "epochs": 200, "lr": args.lr, "scheduler": True, "warmup_epoch": 1,
                    "set_peft": blackbox.FreqFit.set_peft, "set_bb": blackbox.FreqFit.set_bb,
                    "set_training_mode": blackbox.FreqFit.set_training_mode, "pat": 20,
                    "decoder": args.decoder, "bottleneck": args.bottleneck}        
    # Bias parameters tuning
    elif args.method == "Bias":  # ori_lr = 1e-2
        adapt_hp = {"batch_size": 1, "num_samples": 6, "epochs": 200, "lr": args.lr, "scheduler": True, "warmup_epoch": 1,
                    "set_peft": peft.Bias.set_peft, "set_bb": peft.Bias.set_bb,
                    "set_training_mode": peft.Bias.set_training_mode, "pat": 20,
                    "decoder": args.decoder, "bottleneck": args.bottleneck, "freqfit": args.freqfit}
        if args.decoder != "frozen":  # Decrease learning rate if we also update the model decoder
            adapt_hp["lr"] = 1e-4
    # Affine parameters of norm layers tuning
    elif args.method == "Affine":  # ori_lr = 1e-2
        adapt_hp = {"batch_size": 1, "num_samples": 6, "epochs": 200, "lr": args.lr, "scheduler": True, "warmup_epoch": 1,
                    "set_peft": peft.Affine.set_peft, "set_bb": peft.Affine.set_bb,
                    "set_training_mode": peft.Affine.set_training_mode, "pat": 20,
                    "decoder": args.decoder, "bottleneck": args.bottleneck, "freqfit": args.freqfit}
        if args.decoder != "frozen":  # Decrease learning rate if we also update the model decoder
            adapt_hp["lr"] = 1e-4
    # LoRA tuning
    elif args.method == "LoRA": # ori lr = 1e-3
        adapt_hp = {"batch_size": 1, "num_samples": 6, "epochs": 200, "lr": args.lr, "scheduler": True, "warmup_epoch": 1,
                    "rank": 4, "set_peft": peft.LoRA.set_peft, "set_bb": peft.LoRA.set_bb,
                    "set_training_mode": peft.LoRA.set_training_mode, "pat": 20,
                    "decoder": args.decoder, "bottleneck": args.bottleneck, "freqfit": args.freqfit}  # "pat": 40
        if args.decoder != "frozen":  # Decrease learning rate if we also update the model decoder
            adapt_hp["lr"] = 1e-4
    # Compacter tuning
    elif args.method == "Compacter": # ori_lr = 1e-2
        adapt_hp = {"batch_size": 1, "num_samples": 6, "epochs": 200, "lr": args.lr, "scheduler": True, "warmup_epoch": 1,
                    "rank": 4, "set_peft": peft.Compacter.set_peft, "set_bb": peft.Compacter.set_bb,
                    "set_training_mode": peft.Compacter.set_training_mode, "pat": 20,
                    "decoder": args.decoder, "bottleneck": args.bottleneck}
        if args.decoder != "frozen":  # Decrease learning rate if we also update the model decoder
            adapt_hp["lr"] = 1e-4
    # AdaptFormer tuning
    elif args.method == "AdaptFormer":  # ori_lr = 1e-2
        adapt_hp = {"batch_size": 1, "num_samples": 6, "epochs": 200, "lr": args.lr, "scheduler": True, "warmup_epoch": 1,
                    "rank": 4, "set_peft": peft.AdaptFormer.set_peft, "set_bb": peft.AdaptFormer.set_bb,
                    "set_training_mode": peft.AdaptFormer.set_training_mode, "pat": 20,
                    "decoder": args.decoder, "bottleneck": args.bottleneck, "freqfit": args.freqfit}
        if args.decoder != "frozen":  # Decrease learning rate if we also update the model decoder
            adapt_hp["lr"] = 1e-4
    # Residual adapter for CNNs - CNNsAdapter
    elif args.method == "AdapterCNN":  # ori_lr = 1e-2
        adapt_hp = {"batch_size": 1, "num_samples": 6, "epochs": 200, "lr": args.lr, "scheduler": True, "warmup_epoch": 1,
                    "rank": 4, "set_peft": peft.AdapterCNN.set_peft, "set_bb": peft.AdapterCNN.set_bb,
                    "set_training_mode": peft.AdapterCNN.set_training_mode, "pat": 20,
                    "decoder": args.decoder, "bottleneck": args.bottleneck}
        if args.decoder != "frozen":  # Decrease learning rate if we also update the model decoder
            adapt_hp["lr"] = 1e-4
    else:
        print("Tuning strategy not supported...") # ori_lr = 1e-4
        adapt_hp = {"batch_size": 1, "num_samples": 6, "epochs": 200, "lr": args.lr, "scheduler": True, "warmup_epoch": 1,
                    "set_peft": blackbox.LP.set_peft, "set_bb": blackbox.LP.set_bb,
                    "set_training_mode": blackbox.LP.set_training_mode, "pat": 20,
                    "decoder": args.decoder, "bottleneck": args.bottleneck}

    # Parse keys into argparser
    args.adapt_hp = adapt_hp
