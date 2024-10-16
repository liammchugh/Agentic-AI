import cv2
import torch
from torchvision import models, transforms

# Set device to CUDA if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load a pre-trained model for object detection
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model = model.eval().cuda()  # Move the model to GPU

# Define the transformation to be applied to each frame
transform = transforms.Compose([
    transforms.ToTensor()
])

# Initialize the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to a tensor and move it to GPU
    frame_tensor = transform(frame).unsqueeze(0).cuda()

    # Perform object detection
    with torch.no_grad():
        predictions = model(frame_tensor)

    # Process the predictions
    for element in predictions[0]['boxes']:
        x1, y1, x2, y2 = element
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Object Detection', frame)

    # Calculate the distance from the camera for each detected object
    for box in predictions[0]['boxes']:
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        # Assuming a known width of the object in real life (e.g., 36 cm)
        known_width = 36  # cm
        focal_length = 800  # This is an example value, you need to calibrate your camera to get the actual focal length

        # Calculate the distance from the camera
        distance = (known_width * focal_length) / width

        # Display the distance on the frame
        cv2.putText(frame, f"{distance:.2f} cm", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)





    # Break the loop if 'q' or esc is pressed
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) & 0xFF == 27: # 27 is escape key
        break

cap.release()
cv2.destroyAllWindows()
