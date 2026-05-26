import argparse
import json
import os
import pickle
import sys

import numpy as np
import tensorflow as tf
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
	sys.path.insert(0, ROOT_DIR)

from lstm_kmean.model import build_triplenet, test_step, train_step
from lstm_kmean.utils import load_complete_data
from runtime_utils import ensure_dir, write_json


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
	parser = argparse.ArgumentParser(description="Train the EEG feature extractor.")
	parser.add_argument("--data_root", default="data/b2i_data")
	parser.add_argument("--output_dir", required=True)
	parser.add_argument("--encoder_variant", choices=["lstm", "attn_lstm", "lstm_respool", "lstm_statpool", "msconv_bilstm", "resmsconv_lstm", "resbilstm_lstm"], default="lstm")
	parser.add_argument("--warmstart_lstm_ckpt", default="")
	parser.add_argument("--freeze_warmstart_lstm", type=str2bool, default=False)
	parser.add_argument("--restore_ckpt_path", default="")
	parser.add_argument("--epochs", type=int, default=3000)
	parser.add_argument("--batch_size", type=int, default=256)
	parser.add_argument("--feature_dim", type=int, default=128)
	parser.add_argument("--learning_rate", type=float, default=3e-4)
	parser.add_argument("--checkpoint_every_epochs", type=int, default=10)
	parser.add_argument("--max_to_keep", type=int, default=5000)
	parser.add_argument("--max_steps_per_epoch", type=int, default=0)
	parser.add_argument("--seed", type=int, default=45)
	parser.add_argument("--msconv_branch_filters", type=int, default=32)
	parser.add_argument("--msconv_kernel_sizes", type=str, default="3,5,7")
	parser.add_argument("--msconv_dropout", type=float, default=0.1)
	return parser.parse_args()


def append_jsonl(path, payload):
	with open(path, "a", encoding="utf-8") as file:
		file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_best_checkpoint(output_dir, epoch, val_loss, checkpoint_path):
	write_json(
		os.path.join(output_dir, "best_checkpoint.json"),
		{
			"epoch": int(epoch),
			"val_loss": float(val_loss),
			"checkpoint_path": checkpoint_path,
		},
	)


def main():
	args = parse_args()
	np.random.seed(args.seed)
	tf.random.set_seed(args.seed)
	msconv_kernel_sizes = tuple(int(value) for value in args.msconv_kernel_sizes.split(",") if value)

	with open(os.path.join(args.data_root, "eeg", "image", "data.pkl"), "rb") as file:
		data = pickle.load(file, encoding="latin1")

	train_x = data["x_train"]
	train_y = data["y_train"]
	test_x = data["x_test"]
	test_y = data["y_test"]

	train_batch = load_complete_data(train_x, train_y, batch_size=args.batch_size)
	val_batch = load_complete_data(test_x, test_y, batch_size=args.batch_size)
	steps_per_epoch = int(tf.data.experimental.cardinality(train_batch).numpy())

	model = build_triplenet(
		n_classes=train_y.shape[1],
		n_features=args.feature_dim,
		encoder_variant=args.encoder_variant,
		msconv_branch_filters=args.msconv_branch_filters,
		msconv_kernel_sizes=msconv_kernel_sizes,
		msconv_dropout=args.msconv_dropout,
	)
	# Build variables before optional warm-start so nested/lazy layers can receive weights.
	build_x, _ = next(iter(train_batch))
	_ = model(build_x[:1], training=False)
	opt = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

	if args.encoder_variant in {"lstm_respool", "resmsconv_lstm", "resbilstm_lstm"} and args.warmstart_lstm_ckpt:
		baseline_model = build_triplenet(
			n_classes=train_y.shape[1],
			n_features=args.feature_dim,
			encoder_variant="lstm",
		)
		_ = baseline_model(build_x[:1], training=False)
		baseline_opt = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
		baseline_ckpt = tf.train.Checkpoint(
			step=tf.Variable(1),
			model=baseline_model,
			optimizer=baseline_opt,
		)
		baseline_ckpt.restore(args.warmstart_lstm_ckpt).expect_partial()
		if args.encoder_variant == "lstm_respool":
			for idx in range(2):
				model.encoder[idx].set_weights(baseline_model.encoder[idx].get_weights())
		elif args.encoder_variant == "resmsconv_lstm":
			model.encoder.lstm_1.set_weights(baseline_model.encoder[0].get_weights())
			model.encoder.lstm_2.set_weights(baseline_model.encoder[1].get_weights())
		elif args.encoder_variant == "resbilstm_lstm":
			model.encoder.lstm_1.set_weights(baseline_model.encoder[0].get_weights())
			model.encoder.lstm_2.set_weights(baseline_model.encoder[1].get_weights())
		if args.freeze_warmstart_lstm:
			if args.encoder_variant == "lstm_respool":
				for idx in range(2):
					model.encoder[idx].trainable = False
			else:
				model.encoder.lstm_1.trainable = False
				model.encoder.lstm_2.trainable = False
		print(f"Warm-started {args.encoder_variant} from {args.warmstart_lstm_ckpt}")

	ensure_dir(args.output_dir)
	ckpt_dir = os.path.join(args.output_dir, "ckpt")
	log_path = os.path.join(args.output_dir, "train_log.jsonl")
	best_state_path = os.path.join(args.output_dir, "best_checkpoint.json")
	write_json(
		os.path.join(args.output_dir, "config.json"),
		{
			"data_root": os.path.abspath(args.data_root),
			"output_dir": os.path.abspath(args.output_dir),
			"encoder_variant": args.encoder_variant,
			"epochs": args.epochs,
			"batch_size": args.batch_size,
			"feature_dim": args.feature_dim,
			"learning_rate": args.learning_rate,
			"checkpoint_every_epochs": args.checkpoint_every_epochs,
			"max_to_keep": args.max_to_keep,
			"max_steps_per_epoch": args.max_steps_per_epoch,
			"seed": args.seed,
			"warmstart_lstm_ckpt": args.warmstart_lstm_ckpt,
			"freeze_warmstart_lstm": args.freeze_warmstart_lstm,
			"restore_ckpt_path": args.restore_ckpt_path,
			"msconv_branch_filters": args.msconv_branch_filters,
			"msconv_kernel_sizes": list(msconv_kernel_sizes),
			"msconv_dropout": args.msconv_dropout,
		},
	)

	ckpt = tf.train.Checkpoint(
		step=tf.Variable(1),
		epoch=tf.Variable(0),
		model=model,
		optimizer=opt,
	)
	ckpt_manager = tf.train.CheckpointManager(
		ckpt,
		directory=ckpt_dir,
		max_to_keep=args.max_to_keep,
	)
	if args.restore_ckpt_path:
		ckpt.restore(args.restore_ckpt_path).expect_partial()
		print(f"Restored from explicit checkpoint {args.restore_ckpt_path}")
	elif ckpt_manager.latest_checkpoint:
		ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
		print(f"Restored from {ckpt_manager.latest_checkpoint}")

	best_val_loss = float("inf")
	if os.path.exists(best_state_path):
		with open(best_state_path, "r", encoding="utf-8") as file:
			best_state = json.load(file)
		best_val_loss = float(best_state.get("val_loss", best_val_loss))

	step_value = int(ckpt.step.numpy())
	epoch_value = int(ckpt.epoch.numpy())
	if epoch_value > 0:
		start_epoch = epoch_value
	elif step_value > steps_per_epoch * 2:
		# Legacy checkpoints stored global batch steps, not epochs.
		start_epoch = step_value // steps_per_epoch
	else:
		start_epoch = max(step_value - 1, 0)
	ckpt.epoch.assign(start_epoch)

	for epoch in range(start_epoch, args.epochs):
		train_loss = tf.keras.metrics.Mean()
		val_loss = tf.keras.metrics.Mean()

		tq = tqdm(train_batch, desc=f"Train Epoch {epoch + 1}/{args.epochs}")
		for step, (X, Y) in enumerate(tq, start=1):
			if args.max_steps_per_epoch and step > args.max_steps_per_epoch:
				break
			loss = train_step(model, opt, X, Y)
			train_loss.update_state(loss)
			ckpt.step.assign_add(1)
			tq.set_description(
				f"Train Epoch {epoch + 1}/{args.epochs} loss={train_loss.result():0.4f}"
			)

		tq = tqdm(val_batch, desc=f"Val Epoch {epoch + 1}/{args.epochs}")
		for step, (X, Y) in enumerate(tq, start=1):
			if args.max_steps_per_epoch and step > args.max_steps_per_epoch:
				break
			loss = test_step(model, X, Y)
			val_loss.update_state(loss)
			tq.set_description(
				f"Val Epoch {epoch + 1}/{args.epochs} loss={val_loss.result():0.4f}"
			)

		ckpt.epoch.assign(epoch + 1)
		payload = {
			"epoch": epoch + 1,
			"step": int(ckpt.step.numpy()),
			"train_loss": float(train_loss.result().numpy()),
			"val_loss": float(val_loss.result().numpy()),
			"encoder_variant": args.encoder_variant,
		}
		append_jsonl(log_path, payload)
		print(payload)

		saved_path = ""
		if args.checkpoint_every_epochs and (epoch + 1) % args.checkpoint_every_epochs == 0:
			saved_path = ckpt_manager.save(checkpoint_number=epoch + 1)
			print(f"Saved checkpoint: {saved_path}")

		current_val_loss = float(val_loss.result().numpy())
		if current_val_loss < best_val_loss:
			if not saved_path:
				saved_path = ckpt_manager.save(checkpoint_number=epoch + 1)
				print(f"Saved checkpoint: {saved_path}")
			best_val_loss = current_val_loss
			write_best_checkpoint(
				args.output_dir,
				epoch=epoch + 1,
				val_loss=best_val_loss,
				checkpoint_path=saved_path,
			)
			print(
				f"Best checkpoint updated: epoch={epoch + 1}, "
				f"val_loss={best_val_loss:0.6f}, path={saved_path}"
			)

	final_epoch = int(ckpt.epoch.numpy())
	final_path = ckpt_manager.save(checkpoint_number=final_epoch)
	print(f"Final checkpoint: {final_path}")


if __name__ == "__main__":
	main()
