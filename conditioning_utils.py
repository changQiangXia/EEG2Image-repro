import numpy as np
import tensorflow as tf


def select_condition_vector(embedding, feat, condition_source, condition_scale):
    if condition_source == "feat":
        condition = feat
    elif condition_source == "embedding":
        condition = embedding
    elif condition_source == "feat_l2norm":
        condition = tf.nn.l2_normalize(feat, axis=-1)
    else:
        raise ValueError(f"Unsupported condition source: {condition_source}")
    return condition * tf.cast(condition_scale, condition.dtype)


def compute_class_prototypes(
    feature_dataset,
    triplenet,
    n_classes,
    condition_source,
    condition_scale,
):
    prototype_sums = None
    prototype_counts = np.zeros((n_classes,), dtype=np.int64)

    for eeg_batch, label_batch in feature_dataset:
        embedding, features = triplenet(eeg_batch, training=False)
        condition = select_condition_vector(
            embedding,
            features,
            condition_source=condition_source,
            condition_scale=condition_scale,
        ).numpy().astype(np.float32)
        labels = label_batch.numpy().astype(np.int32)
        if prototype_sums is None:
            prototype_sums = np.zeros((n_classes, condition.shape[1]), dtype=np.float32)
        np.add.at(prototype_sums, labels, condition)
        np.add.at(prototype_counts, labels, 1)

    if prototype_sums is None:
        raise ValueError("No features available for prototype computation")
    if np.any(prototype_counts == 0):
        missing = np.where(prototype_counts == 0)[0].tolist()
        raise ValueError(f"Missing prototype samples for classes: {missing}")

    prototypes = prototype_sums / prototype_counts[:, None].astype(np.float32)
    return tf.convert_to_tensor(prototypes, dtype=tf.float32)


def compose_condition_vector(
    condition,
    labels,
    n_classes,
    condition_strategy="direct",
    class_prototypes=None,
    prototype_alpha=1.0,
    post_mix_l2norm=False,
    use_label_condition=False,
):
    labels = tf.cast(labels, tf.int32)
    mixed = condition

    if condition_strategy == "prototype_residual":
        if class_prototypes is None:
            raise ValueError("class_prototypes is required for prototype_residual strategy")
        prototypes = tf.gather(class_prototypes, labels)
        mixed = prototypes + tf.cast(prototype_alpha, condition.dtype) * (condition - prototypes)
    elif condition_strategy != "direct":
        raise ValueError(f"Unsupported condition strategy: {condition_strategy}")

    if post_mix_l2norm:
        mixed = tf.nn.l2_normalize(mixed, axis=-1)

    if use_label_condition:
        label_one_hot = tf.one_hot(labels, depth=n_classes, dtype=mixed.dtype)
        mixed = tf.concat([mixed, label_one_hot], axis=-1)

    return mixed
