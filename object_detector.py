import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

# Initialize the model
model = YOLO("best.pt")

# Streamlit app title
st.title("Object Detection and Analysis App")

# Sidebar for file upload
st.sidebar.title("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Function to detect objects in the image
def detect_objects_on_image(image):
    results = model.predict(image)
    result = results[0]
    output = []
    for box in result.boxes:
        x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)
        output.append([
            x1, y1, x2, y2, result.names[class_id], prob
        ])
    return output

# Function to convert detection results to DataFrame
def convert_to_dataframe(detection_results):
    data = []
    for result in detection_results:
        x1, y1, x2, y2, label, probability = result
        data.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'label': label, 'probability': probability})
    df = pd.DataFrame(data)
    return df

# Process and display image
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    st.write("Detecting objects...")
    detection_results = detect_objects_on_image(image)
    st.write("Detection results:", detection_results)
    
    # Convert results to DataFrame
    df = convert_to_dataframe(detection_results)
    st.write("Detection DataFrame:", df)
    
    # Analysis and Visualization
    bio_degradable = ['plant', 'animal_fish', 'animal_starfish', 'animal_shells',
                      'animal_crab', 'animal_eel', 'animal_etc', 'trash_fabric',
                      'trash_paper', 'trash_rubber', 'trash_wood']
    df['bio_degradable'] = df['label'].apply(lambda x: 1 if x in bio_degradable else 0)

    # Visualization
    counts = df['bio_degradable'].value_counts()
    labels = ['Non-BioDegradable' if i == 0 else 'BioDegradable' for i in counts.index]

    plt.figure(figsize=(10, 5))
    plt.pie(counts, labels=labels, autopct='%1.1f%%', colors=['#FF9999', '#66B2FF'])
    plt.title('Biodegradable vs Non-Biodegradable Waste')
    st.pyplot(plt.gcf())  # Display plot in Streamlit

    # Save plot
    output_file_name = 'uploaded_image_analysis.png'
    plt.savefig(output_file_name)
    plt.close()  # Close the plot to free up memory

    st.success(f"Analysis complete! Plot saved as {output_file_name}")

    # Option to download the analysis image
    with open(output_file_name, "rb") as file:
        btn = st.download_button(
            label="Download Analysis Image",
            data=file,
            file_name=output_file_name,
            mime="image/png"
        )

# Custom HTML component
st.write("### Additional Information")
with open("index.html", "r") as file:
    html_content = file.read()
st.components.v1.html(html_content, height=400)
