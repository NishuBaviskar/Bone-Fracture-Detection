from ultralytics import YOLO

def main():
    # Load the model
    model = YOLO("yolov8n.yaml")  # Replace with your actual model config or weights

    # Path to dataset config
    config_file_path = r"C:/Users/Nishu Baviskar/OneDrive/Desktop/Try Bone Fracture/data.yaml"

    # Project directory and experiment name
    project = r"C:/Users/Nishu Baviskar/OneDrive/Desktop/Try Bone Fracture"
    experiment = "My-Model"

    # Train the model
    result = model.train(
        data=config_file_path,
        epochs=1000,
        project=project,
        name=experiment,
        batch=32,
        device="cpu",         # ✅ Use CPU explicitly
        patience=300,
        imgsz=350,
        verbose=True,
        val=True
    )

if __name__ == "__main__":
    main()

