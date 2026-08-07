from pathlib import Path

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
        self.model_path = Path("artifacts/training/trained_model.h5")
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found at {self.model_path}. "
                "Run the training pipeline before prediction."
            )
        self.model = load_model(self.model_path)

    def predict(self):
        test_image = image.load_img(
            self.filename,
            target_size=(224, 224)
        )

        test_image = image.img_to_array(test_image)
        test_image = test_image / 255.0
        test_image = np.expand_dims(test_image, axis=0)

        probabilities = self.model.predict(test_image)[0]
        result = int(np.argmax(probabilities))
        confidence = float(np.max(probabilities))

        if result == 1:
            prediction = "Tumor"
        else:
            prediction = "Normal"

        return [{
            "image": prediction,
            "confidence": confidence,
            "predicted_class": result,
            "probabilities": probabilities.tolist()
        }]
