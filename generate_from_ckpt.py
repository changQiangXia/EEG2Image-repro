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

from lstm_kmean.model import TripleNet
from model import build_gan
from runtime_utils import configure_runtime, ensure_dir, write_json
from utils import load_complete_data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate EEG-conditioned images from a restored GAN checkpoint."
    )
    parser.add_argument("--data_root", default="data/b2i_data")
    parser.add_argument("--output_dir", default="experiments/checkpoint_validation/ckpt_210")
    parser.add_argument("--triplet_ckpt_dir", default="lstm_kmean/experiments/best_ckpt")
    parser.add_argument("--triplet_ckpt_path", default="")
    parser.add_argument("--gan_ckpt_dir", default="experiments/best_ckpt")
    parser.add_argument("--gan_ckpt_path", default="")
    parser.add_argument("--gan_variant", choices=["simple_gan", "dcgan"], default="dcgan")
    parser.add_argument("--reference_split", default="test")
    parser.add_argument("--test_image_count", type=int, default=50000)
    parser.add_argument("--dataset_batch_size", type=int, default=64)
    parser.add_argument("--generate_batch_size", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=128)
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


def extract_feature_bank(test_batch, triplenet, n_classes, latent_dim, target_counts):
    feature_bank = {class_idx: [] for class_idx in range(n_classes)}
    iterator = tqdm(test_batch, desc="Extracting test EEG features")
    for eeg_batch, label_batch, _ in iterator:
        labels = label_batch.numpy().astype(np.int32)
        features = triplenet(eeg_batch, training=False)[1].numpy().astype(np.float32)
        for label_idx, feature in zip(labels, features):
            feature_bank[int(label_idx)].append(np.squeeze(feature))

    expanded_bank = {}
    for class_idx in range(n_classes):
        features = np.array(feature_bank[class_idx], dtype=np.float32)
        if features.size == 0:
            raise ValueError(f"No EEG features collected for class index {class_idx}")
        repeat_factor = int(math.ceil(target_counts[class_idx] / features.shape[0]))
        tiled = np.expand_dims(features, axis=1)
        tiled = np.tile(tiled, [1, repeat_factor, 1])
        tiled = np.reshape(tiled, [-1, latent_dim])[: target_counts[class_idx]]
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
            generated = model.gen(batch_latent, training=False).numpy()
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

    class_names = discover_class_names(args.data_root)
    n_classes = len(class_names)
    image_dict = build_image_dict(args.data_root, args.reference_split, class_names)

    with open(os.path.join(args.data_root, "eeg", "image", "data.pkl"), "rb") as file:
        data = pickle.load(file, encoding="latin1")

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

    triplenet = TripleNet(n_classes=n_classes)
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

    feature_bank = extract_feature_bank(
        test_batch,
        triplenet,
        n_classes=n_classes,
        latent_dim=args.latent_dim,
        target_counts=target_counts,
    )

    model = build_gan(args.gan_variant)
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
            "gan_ckpt_dir": os.path.abspath(args.gan_ckpt_dir),
            "gan_ckpt_path": gan_path,
            "gan_variant": args.gan_variant,
            "reference_split": args.reference_split,
            "latent_dim": args.latent_dim,
            "test_image_count": args.test_image_count,
            "dataset_batch_size": args.dataset_batch_size,
            "generate_batch_size": args.generate_batch_size,
            "seed": args.seed,
            "class_names": class_names,
            "target_counts": target_counts,
            "train_label_counts": train_label_counts,
            "test_label_counts": test_label_counts,
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
