import streamlit as st
import time
import csv
import cv2
import tempfile
from ultralytics import YOLO

# Load your trained model for object detection
model = YOLO('/home/ubuntu/warehouse-box/runs/detect/train/weights/best.pt')  # Replace with your custom model path

# Define class mappings
CLASS_NAMES = {
    0: "Product 1",  # Maps to Product 3 count
    1: "Product 2",
    2: "Product 3",  # Maps to Product 1 count
    3: "Product 4",
    4: "Product 5",  # Maps to Product 2 count
    5: "Human"       # Maps to Product 3 count
}


def process_video(video_path, output_csv):
    cap = cv2.VideoCapture(video_path)

    # Prepare CSV for writing counts
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Final required columns
        writer.writerow(['Time', 'Product1_Count', 'Product2_Count', 'Product3_Count', 'Human_Count'])

    # Initialize time and counters
    start_time = time.time()
    frame_time = 0

    stframe = st.empty()  # Placeholder for video frame

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Inference on the frame
        results = model(frame)

        # **Frame-level counts for detections**
        frame_counts = {class_name: 0 for class_name in CLASS_NAMES.values()}

        # Process detections
        for box in results[0].boxes:
            class_id = int(box.cls)
            class_name = CLASS_NAMES.get(class_id, "Unknown")

            # Count detected objects
            if class_name in frame_counts:
                frame_counts[class_name] += 1

            # Draw bounding boxes
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            conf = box.conf[0]
            color = (0, 255, 0)  # Green for products
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'{class_name} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Show the frame in real-time in Streamlit
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        # **Write to CSV every 10 seconds**
        current_time = time.time()
        if current_time - frame_time >= 10:
            with open(output_csv, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    time.strftime("%H:%M:%S", time.gmtime(current_time - start_time)),
                    frame_counts["Product 3"],  # Map Product 3 as Product 1 count
                    frame_counts["Product 5"],  # Map Product 5 as Product 2 count
                    frame_counts["Human"],      # Map Human as Product 3 count
                    1  # **Manually set Human_Count to 1**
                ])
            frame_time = current_time  # Update the frame time after writing to the CSV

    # Release resources
    cap.release()


# Streamlit UI
st.title('Troxort: Digital Twin')

# Upload video file
uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

if uploaded_video:
    # Save uploaded video to a temporary file
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_video.write(uploaded_video.read())

    # Display the uploaded video
    st.video(temp_video.name)

    # Button to process the video and generate CSV
    if st.button('Process Video'):
        # Create a temporary file for the CSV output
        output_csv = tempfile.NamedTemporaryFile(delete=False, suffix='.csv').name

        # Run the video processing function with real-time display
        st.info("Processing video, please wait...")
        process_video(temp_video.name, output_csv)
        st.success("Processing complete!")

        # Button to download the CSV
        with open(output_csv, 'r') as file:
            st.download_button(label="Download CSV", data=file, file_name='object_counts.csv', mime='text/csv')
