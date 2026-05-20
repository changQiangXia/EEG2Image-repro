import argparse
import json
import os
import pickle
from glob import glob

import numpy as np
import tensorflow as tf
from natsort import natsorted
from tqdm import tqdm

from lstm_kmean.model import TripleNet
from model import DCGAN, dist_train_step
from runtime_utils import build_strategy, ensure_dir, write_json
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


def save_sample_grid(model, latent, labels, sample_dir, tag):
    ensure_dir(sample_dir)
    generated = model.gen(latent, training=False)
    show_batch_images(generated, os.path.join(sample_dir, f"{tag}.png"), Y=labels)


def append_jsonl(path, payload):
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

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

    triplenet = TripleNet(n_classes=n_classes)
    triplet_opt = tf.keras.optimizers.Adam(learning_rate=3e-4)
    triplet_ckpt = tf.train.Checkpoint(step=tf.Variable(1), model=triplenet, optimizer=triplet_opt)
    triplet_manager = tf.train.CheckpointManager(
        triplet_ckpt,
        directory=args.triplet_ckpt_dir,
        max_to_keep=1,
    )
    triplet_path = resolve_triplet_checkpoint(triplet_manager, args.triplet_ckpt_path)
    triplet_ckpt.restore(triplet_path).expect_partial()

    _, sample_features = triplenet(sample_eeg, training=False)
    if int(sample_features.shape[0]) < 4:
        raise ValueError("batch_size must be at least 4 so sample grids can be rendered.")
    sample_count = min(args.sample_count, int(sample_features.shape[0]))
    sample_count = max(4, sample_count - (sample_count % 4))
    fixed_noise = tf.random.uniform(
        shape=(sample_count, args.latent_dim),
        minval=-0.2,
        maxval=0.2,
    )
    fixed_latent = tf.concat([fixed_noise, sample_features[:sample_count]], axis=-1)
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
            "class_names": class_names,
            "train_label_counts": np.bincount(np.argmax(train_y, axis=1), minlength=n_classes).tolist(),
            "test_label_counts": np.bincount(np.argmax(test_y, axis=1), minlength=n_classes).tolist(),
            "train_image_counts": {name: len(train_image_dict[name]) for name in class_names},
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
    with strategy.scope():
        model = DCGAN()
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

    start_epoch = int(ckpt.epoch.numpy())
    for epoch in range(start_epoch, args.epochs):
        t_gloss = tf.keras.metrics.Mean()
        t_closs = tf.keras.metrics.Mean()

        tq = tqdm(train_batch, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch_idx, (eeg_batch, _, image_batch) in enumerate(tq, start=1):
            if args.max_steps_per_epoch and batch_idx > args.max_steps_per_epoch:
                break
            _, cond_features = triplenet(eeg_batch, training=False)
            gloss, closs = dist_train_step(
                strategy,
                model,
                model_gopt,
                model_copt,
                image_batch,
                cond_features,
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
            "use_diffaug": args.use_diffaug,
            "use_mode_loss": args.use_mode_loss,
            "mode_loss_weight": args.mode_loss_weight,
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
