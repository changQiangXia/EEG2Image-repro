import argparse
import math
import os
import pickle
from glob import glob

import cv2
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
from model import DCGAN
from runtime_utils import configure_runtime, ensure_dir, write_json
from utils import load_complete_data


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
        description="Generate EEG-conditioned images from a restored GAN checkpoint."
    )
    parser.add_argument("--data_root", default="data/b2i_data")
    parser.add_argument("--output_dir", default="experiments/checkpoint_validation/ckpt_210")
    parser.add_argument("--triplet_ckpt_dir", default="lstm_kmean/experiments/best_ckpt")
    parser.add_argument("--triplet_ckpt_path", default="")
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
    parser.add_argument("--gan_ckpt_dir", default="experiments/best_ckpt")
    parser.add_argument("--gan_ckpt_path", default="")
    parser.add_argument("--reference_split", default="test")
    parser.add_argument("--test_image_count", type=int, default=50000)
    parser.add_argument("--dataset_batch_size", type=int, default=64)
    parser.add_argument("--generate_batch_size", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=128)
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


def sample_reference_paths(labels_one_hot, image_dict, class_names, rng):
    sampled_paths = []
    label_indices = np.argmax(labels_one_hot, axis=1)
    for label_idx in label_indices:
        class_name = class_names[int(label_idx)]
        candidates = image_dict[class_name]
        sampled_paths.append(candidates[int(rng.integers(0, len(candidates)))])
    return sampled_paths


def compute_target_counts(total_count, n_classes):
    base = total_count // n_classes
    remainder = total_count % n_classes
    return [base + (1 if class_idx < remainder else 0) for class_idx in range(n_classes)]


def extract_feature_bank(
    test_batch,
    triplenet,
    n_classes,
    target_counts,
    condition_source,
    condition_scale,
    condition_strategy,
    class_prototypes,
    prototype_alpha,
    post_mix_l2norm,
    use_label_condition,
):
    feature_bank = {class_idx: [] for class_idx in range(n_classes)}
    iterator = tqdm(test_batch, desc="Extracting test EEG features")
    for eeg_batch, label_batch, _ in iterator:
        labels = label_batch.numpy().astype(np.int32)
        embedding, features = triplenet(eeg_batch, training=False)
        features = select_condition_vector(
            embedding,
            features,
            condition_source=condition_source,
            condition_scale=condition_scale,
        )
        features = compose_condition_vector(
            features,
            label_batch,
            n_classes=n_classes,
            condition_strategy=condition_strategy,
            class_prototypes=class_prototypes,
            prototype_alpha=prototype_alpha,
            post_mix_l2norm=post_mix_l2norm,
            use_label_condition=use_label_condition,
        ).numpy().astype(np.float32)
        for label_idx, feature in zip(labels, features):
            feature_bank[int(label_idx)].append(np.squeeze(feature))

    expanded_bank = {}
    for class_idx in range(n_classes):
        features = np.array(feature_bank[class_idx], dtype=np.float32)
        if features.size == 0:
            raise ValueError(f"No EEG features collected for class index {class_idx}")
        repeat_factor = int(math.ceil(target_counts[class_idx] / features.shape[0]))
        condition_dim = features.shape[1]
        tiled = np.expand_dims(features, axis=1)
        tiled = np.tile(tiled, [1, repeat_factor, 1])
        tiled = np.reshape(tiled, [-1, condition_dim])[: target_counts[class_idx]]
        expanded_bank[class_idx] = tiled
    return expanded_bank


def resolve_checkpoint_path(manager, explicit_path, label):
    ckpt_path = explicit_path or manager.latest_checkpoint
    if not ckpt_path:
        raise FileNotFoundError(f"No checkpoint found for {label}")
    return ckpt_path


def generate_images(model, class_names, feature_bank, target_counts, output_dir, latent_dim, batch_size, seed):
    image_dir = os.path.join(output_dir, "images")
    ensure_dir(image_dir)
    rng = np.random.default_rng(seed)

    for class_idx, class_name in enumerate(class_names):
        features = feature_bank[class_idx]
        noise = rng.uniform(
            low=-1.0,
            high=1.0,
            size=(target_counts[class_idx], latent_dim),
        ).astype(np.float32)
        latent = np.concatenate([noise, features], axis=-1)
        iterator = range(0, latent.shape[0], batch_size)
        for start in tqdm(iterator, desc=f"Generating {class_name}"):
            batch_latent = tf.convert_to_tensor(latent[start : start + batch_size], dtype=tf.float32)
            batch_size_actual = batch_latent.shape[0]
            batch_labels = tf.fill([batch_size_actual], tf.cast(class_idx, tf.int32))
            generated = model.gen(batch_latent, labels=batch_labels, training=False).numpy()
            for offset, image in enumerate(generated):
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                image = np.uint8(np.clip((image * 0.5 + 0.5) * 255.0, 0, 255))
                image_idx = start + offset
                image_name = f"{class_idx:02d}_{class_name}_{image_idx:05d}.jpg"
                cv2.imwrite(os.path.join(image_dir, image_name), image)


def main():
    args = parse_args()
    configure_runtime()
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    msconv_kernel_sizes = tuple(int(value) for value in args.msconv_kernel_sizes.split(",") if value)

    class_names = discover_class_names(args.data_root)
    n_classes = len(class_names)
    image_dict = build_image_dict(args.data_root, args.reference_split, class_names)

    with open(os.path.join(args.data_root, "eeg", "image", "data.pkl"), "rb") as file:
        data = pickle.load(file, encoding="latin1")

    train_x = data["x_train"]
    train_y = data["y_train"]
    test_x = data["x_test"]
    test_y = data["y_test"]
    train_label_counts = np.bincount(np.argmax(data["y_train"], axis=1), minlength=n_classes).tolist()
    test_label_counts = np.bincount(np.argmax(test_y, axis=1), minlength=n_classes).tolist()
    target_counts = compute_target_counts(args.test_image_count, n_classes)
    reference_paths = sample_reference_paths(test_y, image_dict, class_names, np.random.default_rng(args.seed))
    test_batch = load_complete_data(
        test_x,
        test_y,
        reference_paths,
        batch_size=args.dataset_batch_size,
        dataset_type="eval",
    )

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
    triplet_path = resolve_checkpoint_path(
        triplet_manager,
        args.triplet_ckpt_path,
        "triplet feature extractor",
    )
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

    feature_bank = extract_feature_bank(
        test_batch,
        triplenet,
        n_classes=n_classes,
        target_counts=target_counts,
        condition_source=args.condition_source,
        condition_scale=args.condition_scale,
        condition_strategy=args.condition_strategy,
        class_prototypes=class_prototypes,
        prototype_alpha=args.prototype_alpha,
        post_mix_l2norm=args.post_mix_l2norm,
        use_label_condition=args.use_label_condition,
    )

    condition_dim = int(next(iter(feature_bank.values())).shape[1])
    model = DCGAN(
        n_classes=n_classes,
        gen_input_dim=args.latent_dim + condition_dim,
        gen_label_mode=args.gen_label_mode,
        disc_condition_mode=args.disc_condition_mode,
        disc_label_mode=args.disc_label_mode,
    )
    model_gopt = tf.keras.optimizers.Adam(learning_rate=3e-4, beta_1=0.2, beta_2=0.5)
    model_copt = tf.keras.optimizers.Adam(learning_rate=3e-4, beta_1=0.2, beta_2=0.5)
    gan_ckpt = tf.train.Checkpoint(step=tf.Variable(1), model=model, gopt=model_gopt, copt=model_copt)
    gan_manager = tf.train.CheckpointManager(
        gan_ckpt,
        directory=args.gan_ckpt_dir,
        max_to_keep=1,
    )
    gan_path = resolve_checkpoint_path(gan_manager, args.gan_ckpt_path, "GAN generator")
    gan_ckpt.restore(gan_path).expect_partial()

    ensure_dir(args.output_dir)
    write_json(
        os.path.join(args.output_dir, "config.json"),
        {
            "data_root": os.path.abspath(args.data_root),
            "output_dir": os.path.abspath(args.output_dir),
            "triplet_ckpt_dir": os.path.abspath(args.triplet_ckpt_dir),
            "triplet_ckpt_path": triplet_path,
            "encoder_variant": args.encoder_variant,
            "condition_source": args.condition_source,
            "condition_strategy": args.condition_strategy,
            "condition_scale": args.condition_scale,
            "prototype_alpha": args.prototype_alpha,
            "prototype_batch_size": args.prototype_batch_size,
            "post_mix_l2norm": args.post_mix_l2norm,
            "use_label_condition": args.use_label_condition,
            "gan_ckpt_dir": os.path.abspath(args.gan_ckpt_dir),
            "gan_ckpt_path": gan_path,
            "reference_split": args.reference_split,
            "latent_dim": args.latent_dim,
            "test_image_count": args.test_image_count,
            "dataset_batch_size": args.dataset_batch_size,
            "generate_batch_size": args.generate_batch_size,
            "seed": args.seed,
            "msconv_branch_filters": args.msconv_branch_filters,
            "msconv_kernel_sizes": list(msconv_kernel_sizes),
            "msconv_dropout": args.msconv_dropout,
            "gen_label_mode": args.gen_label_mode,
            "disc_condition_mode": args.disc_condition_mode,
            "disc_label_mode": args.disc_label_mode,
            "class_names": class_names,
            "target_counts": target_counts,
            "train_label_counts": train_label_counts,
            "test_label_counts": test_label_counts,
            "condition_dim": condition_dim,
            "prototype_shape": list(class_prototypes.shape) if class_prototypes is not None else None,
            "reference_image_counts": {
                class_name: len(image_dict[class_name]) for class_name in class_names
            },
        },
    )

    print("Class order:")
    for class_idx, class_name in enumerate(class_names):
        print(
            f"  {class_idx}: {class_name} | "
            f"train labels={train_label_counts[class_idx]} | "
            f"test labels={test_label_counts[class_idx]} | "
            f"reference images={len(image_dict[class_name])} | "
            f"target images={target_counts[class_idx]}"
        )

    generate_images(
        model=model,
        class_names=class_names,
        feature_bank=feature_bank,
        target_counts=target_counts,
        output_dir=args.output_dir,
        latent_dim=args.latent_dim,
        batch_size=args.generate_batch_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
