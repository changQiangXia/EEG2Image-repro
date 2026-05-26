import argparse
import json
import os
import pickle
from glob import glob

import numpy as np
import tensorflow as tf
from natsort import natsorted
from tqdm import tqdm

from conditioning_utils import (
    compose_condition_vector,
    compute_class_prototypes,
    select_condition_vector,
)
from lstm_kmean.model import build_triplenet
from lstm_kmean.utils import load_complete_data as load_feature_data
from model import DCGAN, dist_train_step
from runtime_utils import build_strategy, configure_runtime, ensure_dir, write_json
from utils import load_complete_data, show_batch_images


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the EEG-conditioned GAN with isolated output directories."
    )
    parser.add_argument("--data_root", default="data/b2i_data")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--triplet_ckpt_dir", default="lstm_kmean/experiments/best_ckpt")
    parser.add_argument("--triplet_ckpt_path", default="")
    parser.add_argument("--gan_init_ckpt_dir", default="")
    parser.add_argument("--gan_init_ckpt_path", default="")
    parser.add_argument("--encoder_variant", choices=["lstm", "attn_lstm", "lstm_respool", "lstm_statpool", "msconv_bilstm", "resmsconv_lstm", "resbilstm_lstm"], default="lstm")
    parser.add_argument(
        "--condition_source",
        choices=["feat", "embedding", "feat_l2norm"],
        default="feat",
    )
    parser.add_argument(
        "--condition_strategy",
        choices=["direct", "prototype_residual"],
        default="direct",
    )
    parser.add_argument("--condition_scale", type=float, default=1.0)
    parser.add_argument("--prototype_alpha", type=float, default=1.0)
    parser.add_argument("--prototype_batch_size", type=int, default=256)
    parser.add_argument("--post_mix_l2norm", type=str2bool, default=False)
    parser.add_argument("--use_label_condition", type=str2bool, default=False)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--sample_count", type=int, default=16)
    parser.add_argument("--sample_every_steps", type=int, default=0)
    parser.add_argument("--sample_every_epochs", type=int, default=10)
    parser.add_argument("--checkpoint_every_epochs", type=int, default=10)
    parser.add_argument("--max_to_keep", type=int, default=50)
    parser.add_argument("--max_steps_per_epoch", type=int, default=0)
    parser.add_argument("--use_diffaug", type=str2bool, default=True)
    parser.add_argument("--use_mode_loss", type=str2bool, default=True)
    parser.add_argument("--mode_loss_weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--msconv_branch_filters", type=int, default=32)
    parser.add_argument("--msconv_kernel_sizes", type=str, default="3,5,7")
    parser.add_argument("--msconv_dropout", type=float, default=0.1)
    parser.add_argument(
        "--gen_label_mode",
        choices=["none", "latent_bias"],
        default="none",
    )
    parser.add_argument(
        "--disc_condition_mode",
        choices=["concat", "concat_proj"],
        default="concat",
    )
    parser.add_argument(
        "--disc_label_mode",
        choices=["none", "projection"],
        default="none",
    )
    return parser.parse_args()


def discover_class_names(data_root):
    class_dirs = natsorted(glob(os.path.join(data_root, "images", "train", "*")))
    if not class_dirs:
        raise FileNotFoundError(f"No class folders found under {data_root}/images/train")
    return [os.path.basename(path) for path in class_dirs]


def build_image_dict(data_root, split, class_names):
    image_dict = {}
    for class_name in class_names:
        image_paths = natsorted(glob(os.path.join(data_root, "images", split, class_name, "*")))
        if not image_paths:
            raise FileNotFoundError(
                f"No images found for class '{class_name}' under split '{split}'"
            )
        image_dict[class_name] = image_paths
    return image_dict


def sample_paths(labels_one_hot, image_dict, class_names, rng):
    sampled_paths = []
    label_indices = np.argmax(labels_one_hot, axis=1)
    for label_idx in label_indices:
        class_name = class_names[int(label_idx)]
        candidates = image_dict[class_name]
        sampled_paths.append(candidates[int(rng.integers(0, len(candidates)))])
    return sampled_paths


def resolve_triplet_checkpoint(manager, explicit_path):
    ckpt_path = explicit_path or manager.latest_checkpoint
    if not ckpt_path:
        raise FileNotFoundError("No triplet feature extractor checkpoint found")
    return ckpt_path


def resolve_optional_checkpoint(manager, explicit_path):
    if explicit_path:
        return explicit_path
    if manager is None:
        return ""
    return manager.latest_checkpoint


def save_sample_grid(model, latent, labels, sample_dir, tag):
    ensure_dir(sample_dir)
    label_tensor = tf.convert_to_tensor(labels, dtype=tf.int32)
    generated = model.gen(latent, labels=label_tensor, training=False)
    show_batch_images(generated, os.path.join(sample_dir, f"{tag}.png"), Y=labels)


def append_jsonl(path, payload):
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    configure_runtime()
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    msconv_kernel_sizes = tuple(int(value) for value in args.msconv_kernel_sizes.split(",") if value)

    class_names = discover_class_names(args.data_root)
    n_classes = len(class_names)
    train_image_dict = build_image_dict(args.data_root, "train", class_names)

    with open(os.path.join(args.data_root, "eeg", "image", "data.pkl"), "rb") as file:
        data = pickle.load(file, encoding="latin1")

    train_x = data["x_train"]
    train_y = data["y_train"]
    test_x = data["x_test"]
    test_y = data["y_test"]
    train_paths = sample_paths(train_y, train_image_dict, class_names, rng)

    train_batch = load_complete_data(
        train_x,
        train_y,
        train_paths,
        batch_size=args.batch_size,
        dataset_type="train",
    )

    sample_batch = next(iter(train_batch))
    sample_eeg, sample_labels, _ = sample_batch

    triplenet = build_triplenet(
        n_classes=n_classes,
        n_features=args.latent_dim,
        encoder_variant=args.encoder_variant,
        msconv_branch_filters=args.msconv_branch_filters,
        msconv_kernel_sizes=msconv_kernel_sizes,
        msconv_dropout=args.msconv_dropout,
    )
    triplet_opt = tf.keras.optimizers.Adam(learning_rate=3e-4)
    triplet_ckpt = tf.train.Checkpoint(step=tf.Variable(1), model=triplenet, optimizer=triplet_opt)
    triplet_manager = tf.train.CheckpointManager(
        triplet_ckpt,
        directory=args.triplet_ckpt_dir,
        max_to_keep=1,
    )
    triplet_path = resolve_triplet_checkpoint(triplet_manager, args.triplet_ckpt_path)
    triplet_ckpt.restore(triplet_path).expect_partial()

    class_prototypes = None
    if args.condition_strategy == "prototype_residual":
        prototype_batch = load_feature_data(
            train_x,
            train_y,
            batch_size=args.prototype_batch_size,
        )
        class_prototypes = compute_class_prototypes(
            prototype_batch,
            triplenet,
            n_classes=n_classes,
            condition_source=args.condition_source,
            condition_scale=args.condition_scale,
        )

    sample_embedding, sample_features = triplenet(sample_eeg, training=False)
    sample_condition_base = select_condition_vector(
        sample_embedding,
        sample_features,
        condition_source=args.condition_source,
        condition_scale=args.condition_scale,
    )
    sample_condition = compose_condition_vector(
        sample_condition_base,
        sample_labels,
        n_classes=n_classes,
        condition_strategy=args.condition_strategy,
        class_prototypes=class_prototypes,
        prototype_alpha=args.prototype_alpha,
        post_mix_l2norm=args.post_mix_l2norm,
        use_label_condition=args.use_label_condition,
    )
    if int(sample_features.shape[0]) < 4:
        raise ValueError("batch_size must be at least 4 so sample grids can be rendered.")
    sample_count = min(args.sample_count, int(sample_condition.shape[0]))
    sample_count = max(4, sample_count - (sample_count % 4))
    fixed_noise = tf.random.uniform(
        shape=(sample_count, args.latent_dim),
        minval=-0.2,
        maxval=0.2,
    )
    fixed_latent = tf.concat([fixed_noise, sample_condition[:sample_count]], axis=-1)
    fixed_labels = sample_labels[:sample_count].numpy()

    ensure_dir(args.output_dir)
    ckpt_dir = os.path.join(args.output_dir, "ckpt")
    sample_dir = os.path.join(args.output_dir, "samples")
    log_path = os.path.join(args.output_dir, "train_log.jsonl")
    write_json(
        os.path.join(args.output_dir, "config.json"),
        {
            "data_root": os.path.abspath(args.data_root),
            "output_dir": os.path.abspath(args.output_dir),
            "triplet_ckpt_dir": os.path.abspath(args.triplet_ckpt_dir),
            "triplet_ckpt_path": triplet_path,
            "gan_init_ckpt_dir": os.path.abspath(args.gan_init_ckpt_dir) if args.gan_init_ckpt_dir else "",
            "gan_init_ckpt_path": args.gan_init_ckpt_path,
            "encoder_variant": args.encoder_variant,
            "condition_source": args.condition_source,
            "condition_strategy": args.condition_strategy,
            "condition_scale": args.condition_scale,
            "prototype_alpha": args.prototype_alpha,
            "prototype_batch_size": args.prototype_batch_size,
            "post_mix_l2norm": args.post_mix_l2norm,
            "use_label_condition": args.use_label_condition,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "latent_dim": args.latent_dim,
            "learning_rate": args.learning_rate,
            "sample_count": sample_count,
            "sample_every_steps": args.sample_every_steps,
            "sample_every_epochs": args.sample_every_epochs,
            "checkpoint_every_epochs": args.checkpoint_every_epochs,
            "max_to_keep": args.max_to_keep,
            "max_steps_per_epoch": args.max_steps_per_epoch,
            "use_diffaug": args.use_diffaug,
            "use_mode_loss": args.use_mode_loss,
            "mode_loss_weight": args.mode_loss_weight,
            "seed": args.seed,
            "msconv_branch_filters": args.msconv_branch_filters,
            "msconv_kernel_sizes": list(msconv_kernel_sizes),
            "msconv_dropout": args.msconv_dropout,
            "gen_label_mode": args.gen_label_mode,
            "disc_condition_mode": args.disc_condition_mode,
            "disc_label_mode": args.disc_label_mode,
            "class_names": class_names,
            "train_label_counts": np.bincount(np.argmax(train_y, axis=1), minlength=n_classes).tolist(),
            "test_label_counts": np.bincount(np.argmax(test_y, axis=1), minlength=n_classes).tolist(),
            "train_image_counts": {name: len(train_image_dict[name]) for name in class_names},
            "condition_dim": int(sample_condition.shape[-1]),
            "prototype_shape": list(class_prototypes.shape) if class_prototypes is not None else None,
        },
    )

    print("Class order:")
    for class_idx, class_name in enumerate(class_names):
        print(
            f"  {class_idx}: {class_name} | "
            f"train labels={int(np.sum(np.argmax(train_y, axis=1) == class_idx))} | "
            f"test labels={int(np.sum(np.argmax(test_y, axis=1) == class_idx))} | "
            f"train images={len(train_image_dict[class_name])}"
        )

    strategy = build_strategy()
    gen_input_dim = args.latent_dim + int(sample_condition.shape[-1])
    with strategy.scope():
        model = DCGAN(
            n_classes=n_classes,
            gen_input_dim=gen_input_dim,
            gen_label_mode=args.gen_label_mode,
            disc_condition_mode=args.disc_condition_mode,
            disc_label_mode=args.disc_label_mode,
        )
        model_gopt = tf.keras.optimizers.Adam(
            learning_rate=args.learning_rate,
            beta_1=0.2,
            beta_2=0.5,
        )
        model_copt = tf.keras.optimizers.Adam(
            learning_rate=args.learning_rate,
            beta_1=0.2,
            beta_2=0.5,
        )
        ckpt = tf.train.Checkpoint(
            step=tf.Variable(0),
            epoch=tf.Variable(0),
            model=model,
            gopt=model_gopt,
            copt=model_copt,
        )
        ckpt_manager = tf.train.CheckpointManager(
            ckpt,
            directory=ckpt_dir,
            max_to_keep=args.max_to_keep,
        )
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
            print(f"Resumed from {ckpt_manager.latest_checkpoint}")
        else:
            init_manager = None
            init_path = None
            if args.gan_init_ckpt_dir:
                init_manager = tf.train.CheckpointManager(
                    ckpt,
                    directory=args.gan_init_ckpt_dir,
                    max_to_keep=1,
                )
            init_path = resolve_optional_checkpoint(init_manager, args.gan_init_ckpt_path)
            if init_path:
                ckpt.restore(init_path).expect_partial()
                ckpt.step.assign(0)
                ckpt.epoch.assign(0)
                print(f"Initialized GAN weights from {init_path}")

    start_epoch = int(ckpt.epoch.numpy())
    for epoch in range(start_epoch, args.epochs):
        t_gloss = tf.keras.metrics.Mean()
        t_closs = tf.keras.metrics.Mean()

        tq = tqdm(train_batch, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch_idx, (eeg_batch, label_batch, image_batch) in enumerate(tq, start=1):
            if args.max_steps_per_epoch and batch_idx > args.max_steps_per_epoch:
                break
            cond_embedding, cond_features = triplenet(eeg_batch, training=False)
            cond_vector_base = select_condition_vector(
                cond_embedding,
                cond_features,
                condition_source=args.condition_source,
                condition_scale=args.condition_scale,
            )
            cond_vector = compose_condition_vector(
                cond_vector_base,
                label_batch,
                n_classes=n_classes,
                condition_strategy=args.condition_strategy,
                class_prototypes=class_prototypes,
                prototype_alpha=args.prototype_alpha,
                post_mix_l2norm=args.post_mix_l2norm,
                use_label_condition=args.use_label_condition,
            )
            gloss, closs = dist_train_step(
                strategy,
                model,
                model_gopt,
                model_copt,
                image_batch,
                cond_vector,
                label_batch,
                latent_dim=args.latent_dim,
                batch_size=int(image_batch.shape[0]),
                use_diffaug=args.use_diffaug,
                use_mode_loss=args.use_mode_loss,
                mode_loss_weight=args.mode_loss_weight,
            )
            gloss = tf.reduce_mean(gloss)
            closs = tf.reduce_mean(closs)
            t_gloss.update_state(gloss)
            t_closs.update_state(closs)
            ckpt.step.assign_add(1)

            current_step = int(ckpt.step.numpy())
            if args.sample_every_steps and current_step % args.sample_every_steps == 0:
                save_sample_grid(model, fixed_latent, fixed_labels, sample_dir, f"step_{current_step:07d}")

            tq.set_description(
                f"Epoch {epoch + 1}/{args.epochs} gl={t_gloss.result():0.3f} cl={t_closs.result():0.3f}"
            )

        ckpt.epoch.assign(epoch + 1)
        epoch_index = epoch + 1
        log_payload = {
            "epoch": epoch_index,
            "step": int(ckpt.step.numpy()),
            "generator_loss": float(t_gloss.result().numpy()),
            "discriminator_loss": float(t_closs.result().numpy()),
            "encoder_variant": args.encoder_variant,
            "condition_source": args.condition_source,
            "condition_strategy": args.condition_strategy,
            "condition_scale": args.condition_scale,
            "prototype_alpha": args.prototype_alpha,
            "post_mix_l2norm": args.post_mix_l2norm,
            "use_label_condition": args.use_label_condition,
            "use_diffaug": args.use_diffaug,
            "use_mode_loss": args.use_mode_loss,
            "mode_loss_weight": args.mode_loss_weight,
            "gen_label_mode": args.gen_label_mode,
            "disc_condition_mode": args.disc_condition_mode,
            "disc_label_mode": args.disc_label_mode,
        }
        append_jsonl(log_path, log_payload)
        print(log_payload)

        if args.sample_every_epochs and epoch_index % args.sample_every_epochs == 0:
            save_sample_grid(model, fixed_latent, fixed_labels, sample_dir, f"epoch_{epoch_index:04d}")

        if args.checkpoint_every_epochs and epoch_index % args.checkpoint_every_epochs == 0:
            saved_path = ckpt_manager.save(checkpoint_number=epoch_index)
            print(f"Saved checkpoint: {saved_path}")

    final_epoch = int(ckpt.epoch.numpy())
    final_path = ckpt_manager.save(checkpoint_number=final_epoch)
    print(f"Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
