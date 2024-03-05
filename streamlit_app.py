import streamlit as st
from PIL import Image
import numpy as np
from io import BytesIO
import requests
import pandas as pd
import base64
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest


def preprocess_image_for_clip(image: Image, target_size: int = 224) -> np.ndarray:
    """
    Preprocess the image for the CLIP model.
    - Ensure image is a PIL Image
    - Resize and normalize according to CLIP's expected format
    """
    # Ensure the input is a PIL Image, this step is just for demonstration and can be removed if confirmed
    if not isinstance(image, Image.Image):
        raise ValueError("The provided image is not a PIL Image object")
    
    # Resize the image
    image_resized = image.resize((target_size, target_size))

    # Convert to RGB if not already
    if image_resized.mode != 'RGB':
        image_resized = image_resized.convert('RGB')

    # Convert to NumPy array for any further processing if needed
    image_array = np.array(image_resized)

    # Normalize the image as needed for your model
    # ... normalization steps here

    return image_array


def create_qdrant_client():
    try:
        qdrant_client = QdrantClient(
            url="https://1e1e320e-da3b-42e3-933c-b1558ed8eb60.europe-west3-0.gcp.cloud.qdrant.io:6333",
            api_key="Kap2QDWaKY760_fC_KxInfWMmi0WbG6rSJfXCqrteFRAvzJKRsvMdg",
        )
    except Exception:
        # Docker is unavailable in Google Colab so we switch to local
        # mode available in Python SDK
        qdrant_client = QdrantClient(":memory:")
    
    return qdrant_client

def pillow_image_to_base64(image: Image) -> str:
    """
    Convert a Pillow image to a base64 encoded string that can be used as an image
    source in HTML. If the image has an alpha channel or is in palette mode, it will be converted to 'RGB'.
    :param image: Pillow Image object
    :return: base64 encoded string
    """
    buffered = BytesIO()
    
    # Convert RGBA to RGB if necessary
    if image.mode == 'RGBA' or image.mode == 'P':
        image = image.convert('RGB')
        
    
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_str}"


class ImageProcessor:
    def preprocess_and_encode(self, image):
        model = SentenceTransformer("clip-ViT-B-32")
        image_array = preprocess_image_for_clip(image)
        return model.encode(image)

# Function to generate HTML table
def generate_html_table(data):
    # Start the HTML string for the table
    html = """
    <style>
    .custom-table {
        display: block;
        max-height: 300px;
        overflow-y: auto;
        white-space: nowrap;
    }
    .custom-table th, .custom-table td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: left;
    }
    .custom-table th {
        background-color: #004a7c;
        color: white;
    }
    </style>
    <table class='custom-table'>
    """

    # Add the header row
    html += "<tr>"
    for key in data.keys():
        if "Image" not in key:  # Skip the image key for the header
            html += f"<th>{key}</th>"
    html += "</tr>"

    # Add the data rows
    num_rows = max(len(v) for v in data.values() if isinstance(v, list))  # Find the longest list of values
    for i in range(num_rows):
        html += "<tr>"
        for key, values in data.items():
            if "Image" in key:  # Handle image separately
                continue  # Skip the image data for normal rows
            value = values[i] if i < len(values) else ""
            html += f"<td>{value}</td>"
        html += "</tr>"
        if f"Product {i+1} Image" in data:  # Check if there's an image for this product
            image_url = data[f"Product {i+1} Image"][0]  # Get the image URL
            html += f"<tr><td colspan='{len(data.keys())-1}'><img src='{image_url}' width='100'></td></tr>"
    # Close the table tag
    html += "</table>"
    return html


def reverse_image_search(processor, query_image, limit=2, similarity_threshold =0.5):
    # Preprocess the image and encode it to get the embedding
    query_embedding = processor.preprocess_and_encode(query_image)
    
    # Retrieve search results from Qdrant with the specified limit
    results = create_qdrant_client().search(
        collection_name="product-similarity-search-test-1",
        query_vector=query_embedding,
        with_payload=True,
        limit=limit,
        score_threshold=similarity_threshold,
    )

    # Define the keys you want to extract
    keys = ['variant_title', 'product_title', 'product_vendor', 'product_handle', 'variant_price']

    # Prepare a dictionary to hold your data
    data = {key: [] for key in keys}  # This initializes a list for each key

    # Iterate over each result and collect the data
    for result in results:
        for key in keys:
            # If the key exists in the result payload, append the value to the correct list
            # If the key does not exist, append a None or some default value
            data[key].append(result.payload.get(key, None))

    # Now, convert the dictionary to a DataFrame
    # The keys become the column headers, and the lists become the columns
    df = pd.DataFrame(data)

    # Transpose the DataFrame to have keys in the first column and product details in the subsequent columns
    df_transposed = df.transpose()

    # You now need to make sure that the list you're assigning to df_transposed.columns
    # has the same length as the number of columns in df_transposed.
    # Assuming df initially had the same number of columns as the number of keys
    new_columns = [f'Product {i+1}' for i in range(df_transposed.shape[1])]

    # Assign the new columns to the DataFrame
    df_transposed.columns = new_columns

    # Display the HTML in the Streamlit app
    st.markdown(generate_html_table(df_transposed), unsafe_allow_html=True)

def main():
    # Set Streamlit page configuration
    st.set_page_config(page_title="Manobo Market Intelligence")

    # Sidebar controls
    with st.sidebar:
        st.title("Controls")
        processor = ImageProcessor()
        uploaded_file = st.file_uploader("Upload a product image:", type=['png', 'jpg', 'jpeg'])
        
        # Let the user decide how many similar images to retrieve
        num_similar_images = st.number_input("Number of similar images to retrieve", min_value=1, value=2, step=1)

        # Let the user decide how many similar images to retrieve
        similarity_threshold = st.number_input("Similarity threshold", min_value=0.0, value=0.5, step=0.01, max_value=1.0)

        search_button = st.button('Search')

    # Main area for displaying results
    st.title("Manobo Style Sage")

    # When the 'Search' button is pressed and an image is provided, perform the search
    if search_button and uploaded_file:
        query_image = Image.open(uploaded_file)
        reverse_image_search(processor, query_image, limit=num_similar_images, similarity_threshold=similarity_threshold)

if __name__ == "__main__":
    main()
