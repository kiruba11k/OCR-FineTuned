import streamlit as st
from transformers import AutoProcessor, AutoModelForTokenClassification
from PIL import Image
import torch
import pandas as pd
import pytesseract

# Load fine-tuned model and processor
processor = AutoProcessor.from_pretrained("Kiruba11/layoutlmv3-resume-ner2")
model = AutoModelForTokenClassification.from_pretrained("Kiruba11/layoutlmv3-resume-ner2")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Helper function to extract entities

def predict_entities(image: Image.Image):
    # Use Tesseract to extract words and bounding boxes
    ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    words = []
    boxes = []

    for i in range(len(ocr_data["text"])):
        word = ocr_data["text"][i]
        if word.strip() == "":
            continue
        x, y, w, h = ocr_data["left"][i], ocr_data["top"][i], ocr_data["width"][i], ocr_data["height"][i]
        words.append(word)
        boxes.append([x, y, x + w, y + h])

    # Resize boxes to 0–1000 scale as required by LayoutLMv3
    width, height = image.size
    normalized_boxes = [
        [
            int(1000 * (box[0] / width)),
            int(1000 * (box[1] / height)),
            int(1000 * (box[2] / width)),
            int(1000 * (box[3] / height)),
        ]
        for box in boxes
    ]

    # Processor expects words and boxes
    encoding = processor(image=image, words=words, boxes=normalized_boxes, return_tensors="pt", truncation=True)
    encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = model(**encoding)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)

    tokens = processor.tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
    labels = [model.config.id2label[p.item()] for p in predictions[0]]

    results = []
    for token, label in zip(tokens, labels):
        if token not in processor.tokenizer.all_special_tokens and label != 'O':
            results.append((token.replace('▁', ''), label))

    return results


# Streamlit UI
st.set_page_config(page_title="OCR Entity Extractor", layout="wide")
st.title("Image OCR ")
st.markdown("Upload one or more images to extract.")

uploaded_files = st.file_uploader("Upload image files (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    all_results = []

    for file in uploaded_files:
        st.image(file, caption=f"Uploaded: {file.name}", use_column_width=True)
        image = Image.open(file).convert("RGB")
        entities = predict_entities(image)

        # Combine same entity types
        combined = {}
        for word, label in entities:
            key = label[2:] if "-" in label else label
            combined[key] = combined.get(key, "") + word + " "

        combined["Image Name"] = file.name
        all_results.append(combined)

    # Create dataframe
    df = pd.DataFrame(all_results)

    st.success("Extraction complete.")
    st.dataframe(df)

    # Download CSV
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "extracted_entities.csv", "text/csv")
else:
    st.info("Please upload at least one image to start.")
