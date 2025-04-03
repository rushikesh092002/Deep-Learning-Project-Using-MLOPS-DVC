import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from pathlib import Path
from cnnClassifier.entity.config_entity import TrainingConfig

class Training:
    def __init__(self, config:TrainingConfig):
        self.config = config
        

    def get_base_model(self):
        self.model = tf.keras.models.load_model(self.config.updated_base_model_path)

        # ✅ Fix: Reinitialize the optimizer with the correct learning rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.params_learning_rate)

        # ✅ Fix: Recompile the model with the new optimizer
        self.model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",  # Change if needed
            metrics=["accuracy"]
        )

    def train_valid_generator(self):
        """Creates training and validation data generators."""
        datagenerator_kwargs = dict(
            rescale=1. / 255,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

        # ✅ Debugging: Check if generators work properly
        batch = next(iter(self.train_generator))
        print("Batch type:", type(batch))  # Should be tuple (images, labels)
        print("Image batch shape:", batch[0].shape)  # Check image shape
        print("Label batch shape:", batch[1].shape)  # Check label shape

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """Saves the trained model."""
        model.save(path)

    def train(self, callback_list: list):
        """Trains the model using generators."""
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator,
            callbacks=callback_list
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )