import os
import numpy as np
from PIL import Image
import pickle


class FaceRecognizer:
    def __init__(self):
        self.model_weights = None
        self.label_names = ["Elon Musk", "Cristiano Ronaldo", "Unknown"]
        self.base_path = os.path.join(os.path.dirname(__file__), 'data')

    def load_images(self, folder_name, label_index):
        images = []
        labels = []
        folder_path = os.path.join(self.base_path, folder_name)

        for filename in sorted(os.listdir(folder_path)):
            if filename.lower().endswith(('.jpg')):
                try:
                    img_path = os.path.join(folder_path, filename)
                    img = Image.open(img_path)
                    img = img.convert('L').resize((50, 50))
                    images.append(np.array(img).flatten())
                    labels.append(label_index)
                except Exception as e:
                    print(f"Skipped {filename}: {str(e)}")
        return images, labels

    def train(self):
        elon, e_labels = self.load_images("elon", 0)
        ronaldo, r_labels = self.load_images("ronaldo", 1)
        unknown, u_labels = self.load_images("unknown", 2)

        X = np.array(elon + ronaldo + unknown) / 255.0
        y = np.array(e_labels + r_labels + u_labels)

        X_with_bias = np.c_[np.ones((X.shape[0], 1)), X]
        self.model_weights = np.linalg.pinv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y

    def predict(self, image_path):
        try:
            img = Image.open(image_path).convert('L').resize((50, 50))
            img_array = np.array(img).flatten() / 255.0
            x_with_bias = np.insert(img_array, 0, 1)
            prediction = int(np.round(x_with_bias @ self.model_weights).clip(0, 2))
            return self.label_names[prediction]
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return "Error"

    def save_model(self, filename="face_model.pkl"):
        with open(filename, 'wb') as f:
            pickle.dump({'weights': self.model_weights}, f)

    def load_model(self, filename="face_model.pkl"):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.model_weights = data['weights']


def test_all_images(recognizer, person, count=30):
    print(f"\nTesting {person} images:")
    correct = 0
    for i in range(1, count + 1):
        img_path = os.path.join(recognizer.base_path, person.lower(), f"{person.lower()}{i}.jpg")
        if os.path.exists(img_path):
            prediction = recognizer.predict(img_path)
            expected = "Elon Musk" if person.lower() == "elon" else "Cristiano Ronaldo" if person.lower() == "ronaldo" else "Unknown"
            result = "✓" if prediction == expected else "✗"
            print(f"{person}{i}.jpg: {result}")
            if prediction == expected:
                correct += 1
        else:
            print(f"Image not found: {img_path}")
    accuracy = (correct / count) * 100
    print(f"Accuracy for {person}: {accuracy:.2f}% ({correct}/{count})")
    return accuracy


def main():
    recognizer = FaceRecognizer()

    print("Training model...")
    recognizer.train()
    recognizer.save_model()
    print("Model trained and saved!")

    test_all_images(recognizer, "elon")
    test_all_images(recognizer, "ronaldo")
    test_all_images(recognizer, "unknown")

if __name__ == "__main__":
    main()
