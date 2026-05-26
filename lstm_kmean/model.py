import tensorflow as tf
from tensorflow.keras import Model, layers
import tensorflow_addons as tfa


class AttentionReadout(layers.Layer):
	def __init__(self, hidden_dim):
		super(AttentionReadout, self).__init__()
		self.score_hidden = layers.Dense(hidden_dim, activation="tanh")
		self.score_out = layers.Dense(1, use_bias=False)
		self.gate = layers.Dense(hidden_dim, activation="sigmoid")
		self.norm = layers.LayerNormalization(epsilon=1e-6)

	def call(self, sequence_features):
		scores = self.score_out(self.score_hidden(sequence_features))
		weights = tf.nn.softmax(scores, axis=1)
		context = tf.reduce_sum(sequence_features * weights, axis=1)
		last_state = sequence_features[:, -1, :]
		gate = self.gate(tf.concat([last_state, context], axis=-1))
		return self.norm(gate * last_state + (1.0 - gate) * context)


class AttentionLSTMEncoder(layers.Layer):
	def __init__(self, n_features):
		super(AttentionLSTMEncoder, self).__init__()
		self.lstm_1 = layers.LSTM(units=32, return_sequences=True)
		self.lstm_2 = layers.LSTM(units=n_features, return_sequences=True)
		self.readout = AttentionReadout(hidden_dim=n_features)

	def call(self, x, training=False):
		x = self.lstm_1(x, training=training)
		x = self.lstm_2(x, training=training)
		return self.readout(x)


class LSTMStatPoolEncoder(layers.Layer):
	def __init__(self, n_features):
		super(LSTMStatPoolEncoder, self).__init__()
		self.lstm_1 = layers.LSTM(units=32, return_sequences=True)
		self.lstm_2 = layers.LSTM(units=n_features, return_sequences=True)
		self.stats_hidden = layers.Dense(n_features, activation="gelu")
		self.stats_out = layers.Dense(
			n_features,
			kernel_initializer="zeros",
			bias_initializer="zeros",
		)

	def call(self, x, training=False):
		x = self.lstm_1(x, training=training)
		x = self.lstm_2(x, training=training)
		last = x[:, -1, :]
		mean = tf.reduce_mean(x, axis=1)
		variance = tf.reduce_mean(tf.square(x - mean[:, tf.newaxis, :]), axis=1)
		std = tf.sqrt(variance + 1e-6)
		stats = tf.concat([mean, std], axis=-1)
		delta = self.stats_out(self.stats_hidden(stats))
		return last + delta


class MultiScaleConvStem(layers.Layer):
	def __init__(self, branch_filters=32, kernel_sizes=(3, 5, 7), dropout_rate=0.1):
		super(MultiScaleConvStem, self).__init__()
		self.norm = layers.LayerNormalization(epsilon=1e-6)
		self.branches = []
		for kernel_size in kernel_sizes:
			self.branches.append(
				tf.keras.Sequential(
					[
						layers.Conv1D(
							filters=branch_filters,
							kernel_size=kernel_size,
							padding="same",
							use_bias=False,
						),
						layers.BatchNormalization(),
						layers.Activation("gelu"),
					]
				)
			)
		self.fuse = layers.Dense(branch_filters * len(kernel_sizes))
		self.drop = layers.Dropout(dropout_rate)

	def call(self, x, training=False):
		x = self.norm(x)
		branches = [branch(x, training=training) for branch in self.branches]
		x = tf.concat(branches, axis=-1)
		x = self.fuse(x)
		return self.drop(x, training=training)


class MSConvBiLSTMEncoder(layers.Layer):
	def __init__(
		self,
		n_features,
		branch_filters=32,
		kernel_sizes=(3, 5, 7),
		dropout_rate=0.1,
	):
		super(MSConvBiLSTMEncoder, self).__init__()
		self.stem = MultiScaleConvStem(
			branch_filters=branch_filters,
			kernel_sizes=kernel_sizes,
			dropout_rate=dropout_rate,
		)
		self.bilstm_1 = layers.Bidirectional(
			layers.LSTM(units=64, return_sequences=True, dropout=dropout_rate)
		)
		self.bilstm_2 = layers.Bidirectional(
			layers.LSTM(units=n_features // 2, return_sequences=False, dropout=dropout_rate)
		)

	def call(self, x, training=False):
		x = self.stem(x, training=training)
		x = self.bilstm_1(x, training=training)
		x = self.bilstm_2(x, training=training)
		return x


class ResidualMSConvLSTMEncoder(layers.Layer):
	def __init__(
		self,
		n_features,
		branch_filters=32,
		kernel_sizes=(3, 5, 7),
		dropout_rate=0.1,
	):
		super(ResidualMSConvLSTMEncoder, self).__init__()
		self.stem = MultiScaleConvStem(
			branch_filters=branch_filters,
			kernel_sizes=kernel_sizes,
			dropout_rate=dropout_rate,
		)
		self.residual_proj = None
		self.lstm_1 = layers.LSTM(units=32, return_sequences=True)
		self.lstm_2 = layers.LSTM(units=n_features, return_sequences=False)

	def build(self, input_shape):
		channel_dim = int(input_shape[-1])
		self.residual_proj = layers.Dense(
			units=channel_dim,
			kernel_initializer="zeros",
			bias_initializer="zeros",
		)
		super(ResidualMSConvLSTMEncoder, self).build(input_shape)

	def call(self, x, training=False):
		residual = self.residual_proj(self.stem(x, training=training))
		x = x + residual
		x = self.lstm_1(x, training=training)
		x = self.lstm_2(x, training=training)
		return x


class ResidualBiLSTMEncoder(layers.Layer):
	def __init__(self, n_features, dropout_rate=0.1):
		super(ResidualBiLSTMEncoder, self).__init__()
		self.lstm_1 = layers.LSTM(units=32, return_sequences=True)
		self.lstm_2 = layers.LSTM(units=n_features, return_sequences=False)
		self.backward_lstm_1 = layers.LSTM(
			units=32,
			return_sequences=True,
			go_backwards=True,
			dropout=dropout_rate,
		)
		self.backward_lstm_2 = layers.LSTM(
			units=n_features,
			return_sequences=False,
			go_backwards=True,
			dropout=dropout_rate,
		)
		self.residual_proj = layers.Dense(
			units=n_features,
			kernel_initializer="zeros",
			bias_initializer="zeros",
		)

	def call(self, x, training=False):
		forward_seq = self.lstm_1(x, training=training)
		forward = self.lstm_2(forward_seq, training=training)
		backward_seq = self.backward_lstm_1(x, training=training)
		backward = self.backward_lstm_2(backward_seq, training=training)
		return forward + self.residual_proj(backward)


class TripleNet(Model):
	def __init__(
		self,
		n_classes=10,
		n_features=128,
		encoder_variant="lstm",
		msconv_branch_filters=32,
		msconv_kernel_sizes=(3, 5, 7),
		msconv_dropout=0.1,
	):
		super(TripleNet, self).__init__()
		del n_classes
		self.encoder_variant = encoder_variant
		if encoder_variant == "lstm":
			filters = [32, n_features]
			ret_seq = [True, False]
			self.enc_depth = len(filters)
			self.encoder = [
				layers.LSTM(units=filters[idx], return_sequences=ret_seq[idx])
				for idx in range(self.enc_depth)
			]
		elif encoder_variant == "attn_lstm":
			self.encoder = AttentionLSTMEncoder(n_features=n_features)
		elif encoder_variant == "lstm_respool":
			filters = [32, n_features]
			ret_seq = [True, False]
			self.enc_depth = len(filters)
			self.encoder = [
				layers.LSTM(units=filters[idx], return_sequences=ret_seq[idx])
				for idx in range(self.enc_depth)
			]
			self.respool_hidden = layers.Dense(n_features, activation="gelu")
			self.respool_out = layers.Dense(
				n_features,
				kernel_initializer="zeros",
				bias_initializer="zeros",
			)
		elif encoder_variant == "lstm_statpool":
			self.encoder = LSTMStatPoolEncoder(n_features=n_features)
		elif encoder_variant == "msconv_bilstm":
			self.encoder = MSConvBiLSTMEncoder(
				n_features=n_features,
				branch_filters=msconv_branch_filters,
				kernel_sizes=msconv_kernel_sizes,
				dropout_rate=msconv_dropout,
			)
		elif encoder_variant == "resmsconv_lstm":
			self.encoder = ResidualMSConvLSTMEncoder(
				n_features=n_features,
				branch_filters=msconv_branch_filters,
				kernel_sizes=msconv_kernel_sizes,
				dropout_rate=msconv_dropout,
			)
		elif encoder_variant == "resbilstm_lstm":
			self.encoder = ResidualBiLSTMEncoder(
				n_features=n_features,
				dropout_rate=msconv_dropout,
			)
		else:
			raise ValueError(f"Unsupported encoder variant: {encoder_variant}")
		self.flat = layers.Flatten()

	def call(self, x, training=False):
		if self.encoder_variant == "lstm":
			for idx in range(self.enc_depth):
				x = self.encoder[idx](x, training=training)
		elif self.encoder_variant == "lstm_respool":
			seq = self.encoder[0](x, training=training)
			base = self.encoder[1](seq, training=training)
			mean = tf.reduce_mean(seq, axis=1)
			variance = tf.reduce_mean(tf.square(seq - mean[:, tf.newaxis, :]), axis=1)
			std = tf.sqrt(variance + 1e-6)
			stats = tf.concat([mean, std], axis=-1)
			residual = self.respool_out(self.respool_hidden(stats))
			x = base + residual
		else:
			x = self.encoder(x, training=training)
		feat = self.flat(x)
		x = tf.nn.l2_normalize(feat, axis=-1)
		return x, feat


def build_triplenet(
	n_classes=10,
	n_features=128,
	encoder_variant="lstm",
	msconv_branch_filters=32,
	msconv_kernel_sizes=(3, 5, 7),
	msconv_dropout=0.1,
):
	return TripleNet(
		n_classes=n_classes,
		n_features=n_features,
		encoder_variant=encoder_variant,
		msconv_branch_filters=msconv_branch_filters,
		msconv_kernel_sizes=msconv_kernel_sizes,
		msconv_dropout=msconv_dropout,
	)

@tf.function
def train_step(softnet, opt, X, Y):
	with tf.GradientTape() as tape:
		Y_emb, _ = softnet(X, training=True)
		# loss  = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(Y, Y_emb)
		loss  = tfa.losses.TripletSemiHardLoss()(Y, Y_emb)
	variables = softnet.trainable_variables
	gradients = tape.gradient(loss, variables)
	opt.apply_gradients(zip(gradients, variables))
	return loss

@tf.function
def test_step(softnet, X, Y):
	Y_emb, _ = softnet(X, training=False)
	loss  = tfa.losses.TripletSemiHardLoss()(Y, Y_emb)
	# loss  = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(Y, Y_emb)
	return loss
