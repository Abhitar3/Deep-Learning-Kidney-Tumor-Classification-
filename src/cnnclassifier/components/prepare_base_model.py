import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from cnnclassifier import logger
from cnnclassifier.entity.config_entity import prepareBaseModelConfig
from pathlib import Path


class PrepareBaseModel:
    def __init__(self, config: prepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        try:
            self.model = tf.keras.applications.vgg16.VGG16(
                input_shape=self.config.params_image_size,
                include_top=self.config.params_include_top,
                weights=self.config.params_weights,
                classes=self.config.params_classes
            )
            self.model.save(self.config.base_model_path)
            logger.info(f"Base model is saved at {self.config.base_model_path}")
        except Exception as e:
            logger.error(f"Error occurred while saving base model: {e}")
            raise e

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all=True, freeze_till=None, learning_rate=0.01):
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:freeze_till]:
                layer.trainable = False

        flatten_in = tf.keras.layers.Flatten()(model.output)
        prediction = tf.keras.layers.Dense(units=classes, activation="softmax")(flatten_in)
        full_model = tf.keras.models.Model(inputs=model.input, outputs=prediction)
        full_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        full_model.summary()
        return full_model

    def update_base_model(self):
        self.model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )
        self.model.save(self.config.updated_base_model_path)
