import tensorflow as tf
from pysc2.lib import actions
from pysc2.lib.features import SCREEN_FEATURES, MINIMAP_FEATURES
from tensorflow.contrib import layers


class FullyConvPolicy:
    """
    FullyConv network structure from https://arxiv.org/pdf/1708.04782.pdf
    Some implementation ideas are borrowed from https://github.com/xhujoy/pysc2-agents
    """

    def __init__(self,
            agent,
            trainable: bool = True
    ):
        # type agent: ActorCriticAgent
        self.placeholders = agent.placeholders
        self.trainable = trainable
        self.unittype_emb_dim = agent.unit_type_emb_dim

    def _build_convs(self, inputs, name):
        self.conv1 = layers.conv2d(
            inputs=inputs,
            data_format="NHWC",
            num_outputs=16,
            kernel_size=5,
            stride=1,
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv1" % name,
            trainable=self.trainable
        )
        self.conv2 = layers.conv2d(
            inputs=self.conv1,
            data_format="NHWC",
            num_outputs=32,
            kernel_size=3,
            stride=1,
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv2" % name,
            trainable=self.trainable
        )

        if self.trainable:
            layers.summarize_activation(self.conv1)
            layers.summarize_activation(self.conv2)

        return self.conv2

    def build(self):
        self.units_embedded = layers.embed_sequence(
            self.placeholders.screen_unit_type,
            vocab_size=SCREEN_FEATURES.unit_type.scale,
            embed_dim=self.unittype_emb_dim,
            scope="unit_type_emb",
            trainable=self.trainable
        )

        # Let's not one-hot zero which is background
        self.player_relative_screen_one_hot = layers.one_hot_encoding(
            self.placeholders.player_relative_screen,
            num_classes=SCREEN_FEATURES.player_relative.scale
        )[:, :, :, 1:]
        self.player_relative_minimap_one_hot = layers.one_hot_encoding(
            self.placeholders.player_relative_minimap,
            num_classes=MINIMAP_FEATURES.player_relative.scale
        )[:, :, :, 1:]

        channel_axis = 3
        self.screen_numeric_all = tf.concat(
            [self.placeholders.screen_numeric, self.units_embedded, self.player_relative_screen_one_hot],
            axis=channel_axis
        )
        self.minimap_numeric_all = tf.concat(
            [self.placeholders.minimap_numeric, self.player_relative_minimap_one_hot],
            axis=channel_axis
        )
        self.screen_output = self._build_convs(self.screen_numeric_all, "screen_network")
        self.minimap_output = self._build_convs(self.minimap_numeric_all, "minimap_network")

        self.map_output = tf.concat([self.screen_output, self.minimap_output], axis=channel_axis)

        self.spatial_action_logits = layers.conv2d(
            self.map_output,
            data_format="NHWC",
            num_outputs=1,
            kernel_size=1,
            stride=1,
            activation_fn=None,
            scope='spatial_action',
            trainable=self.trainable
        )

        self.spatial_action_probs = tf.nn.softmax(layers.flatten(self.spatial_action_logits))

        self.map_output_flat = layers.flatten(self.map_output)

        self.fc1 = layers.fully_connected(
            self.map_output_flat,
            num_outputs=256,
            activation_fn=tf.nn.relu,
            scope="fc1",
            trainable=self.trainable
        )
        self.action_id_probs = layers.fully_connected(
            self.fc1,
            num_outputs=len(actions.FUNCTIONS),
            activation_fn=tf.nn.softmax,
            scope="action_id",
            trainable=self.trainable
        )
        self.value_estimate = tf.squeeze(layers.fully_connected(
            self.fc1,
            num_outputs=1,
            activation_fn=None,
            scope='value',
            trainable=self.trainable
        ), axis=1)

        # disregard non-allowed actions by setting zero prob and re-normalizing to 1
        self.action_id_probs *= self.placeholders.available_action_ids
        self.action_id_probs /= tf.reduce_sum(self.action_id_probs, axis=1, keep_dims=True)

        def logclip(x):
            return tf.log(tf.clip_by_value(x, 1e-12, 1.0))

        spatial_action_log_probs = (
            logclip(self.spatial_action_probs)
            * tf.expand_dims(self.placeholders.is_spatial_action_available, axis=1)
        )

        # non-available actions get log(1e-10) value but that's ok because it's never used
        self.action_id_log_probs = logclip(self.action_id_probs)

        self.value_estimate = self.value_estimate
        self.action_id_probs = self.action_id_probs
        self.spatial_action_probs = self.spatial_action_probs
        self.action_id_log_probs = self.action_id_log_probs
        self.spatial_action_log_probs = spatial_action_log_probs
        return self
