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
    """
    generates a Qdrant client instance to connect to a Qdrant endpoint. The endpoint
    is specified through the `url` parameter, while the API key is passed as a
    separate argument. If Docker is unavailable in Google Colab, the function falls
    back to using the local mode available in the Python SDK.

    Returns:
        `QdrantClient` instance.: a QDrant client instance.
        
        		- `qdrant_client`: A QdrantClient object representing a connection to
        the Qdrant platform.
        

    """
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
        """
        takes an input image and applies a preprocessing step using the
        `preprocess_image_for_clip` function, followed by encoding it using a
        sentence transformer model named "clip-ViT-B-32".

        Args:
            image (ndarray.): 4D tensor containing the raw image data that will
                be preprocessed and encoded by the `SentenceTransformer` model.
                
                		- `image`: A PIL Image object containing a 3D numpy array
                representing the input image in the format `(height, width,
                channels)`, where height and width denote the spatial dimensions,
                and channels denotes the number of color channels (typically 3 for
                RGB images).
                

        Returns:
            instance of `torch.Tensor`.: a sequence of numerical representations
            of the input image, obtained through encoding with the SentenceTransformer
            model.
            
            	1/ The output is a tensor with shape (1, batch_size, 32) or (1,
            sequence_length, 32), where batch_size and sequence_length are dependent
            on the input image and are not provided explicitly.
            	2/ Each element in the tensor represents a tokenized and encoded
            representation of the image. The tokens are derived from the original
            image pixels, using a predefined model, and the encoding is performed
            by the SentenceTransformer model.
            	3/ The encoded representation consists of 32 dimensions, where each
            dimension corresponds to a particular aspect of the input image, such
            as texture or color information.
            	4/ The output tensor can be used for various computer vision tasks,
            such as image classification, object detection, and segmentation.
            	5/ The function returns a single tensor with the encoded representation
            of the input image, without any additional features or modifications
            to the input data.
            

        """
        model = SentenceTransformer("clip-ViT-B-32")
        image_array = preprocess_image_for_clip(image)
        return model.encode(image)

def generate_html_table(results, keys):
    # Start the HTML string for the table
    """
    takes in a list of results and a list of keys as input, generates an HTML table
    with columns matching the keys, and appends each result's payload data to a
    row in the table.

    Args:
        results (list): 2D list containing the output of an API call, which is
            then transformed into an HTML table format using the other input
            parameters and a series of string concatenations.
        keys (list): list of column headers or keys

    Returns:
        str: an HTML table with columns for image, price, vendor, title, handle,
        and other custom keys.

    """
    html = """
    <style>
    .custom-table {
        display: block;
        max-height: 600px;  /* Corrected from 600x to 600px */
        max-width: 600px;
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
    .custom-table img {
        max-width: 250px;  /* Adjust the size as necessary */
        max-height: 250px;
    }
    </style>
    <table class='custom-table'>
    <tr>
        <th>Image</th> <!-- Column header for images -->
        <th>variant_price</th>
        <th>product_vendor</th>
        <th>variant_title</th>
        <th>product_title</th>
        <th>product_handle</th>
    </tr>
    """

    # Iterate over each result and create a row in the table
    for result in results:
        payload = result.payload
        html += "<tr>"
        # Add the image cell using the 'variant_featured_image' key
        image_url = payload.get('variant_featured_image', '')
        html += f"<td><img src='{image_url}' alt='Product Image'></td>"

        # Add other cells based on keys
        for key in keys:
            if key != 'variant_featured_image':  # Exclude image URL from table cells
                value = payload.get(key, '')
                html += f"<td>{value}</td>"
        html += "</tr>"

    html += "</table>"
    return html




def reverse_image_search(processor, query_image, limit=2, similarity_threshold =0.5):
    # Preprocess the image and encode it to get the embedding
    """
    takes an image and a limit for searching similar images in Qdrant, preprocesses
    and encodes the image, creates a Qdrant client to retrieve search results
    within the specified limit and with a score threshold, and displays the HTML
    table of search results using Streamlit.

    Args:
        processor (`object`.): third-party object responsible for encoding and
            processing the input image to produce an embedding for search purposes.
            
            		- `processor`: an instance of the `ImageProcessor` class, which
            defines the preprocessing and encoding steps for images. This input
            parameter is expected to contain the necessary configurations and
            attributes for carrying out these processes.
            
            	The function then retrieves search results from Qdrant using a client
            instance created specifically for this purpose. The parameters passed
            to the `search()` method include:
            
            		- `collection_name`: the name of the collection containing the images
            being searched (a string value).
            		- `query_vector`: the query image embedding generated by the `processor`
            (a NumPy array representing a vector of float values).
            		- `with_payload`: a boolean value indicating whether the search
            response should include payload data along with the results (default:
            `True`).
            		- `limit`: an integer value representing the maximum number of search
            results to retrieve (default: `2`).
            		- `score_threshold`: a float value representing the minimum similarity
            score required for an image to be considered a match (default: `0.5`).
            
            	The function then extracts the desired keys from the search response
            and generates an HTML table using the `generate_html_table()` function.
            This table displays the extracted key-value pairs in a human-readable
            format.
            
        query_image (image.): image that the function will use to search for similar
            products on Qdrant, and it is preprocessed and encoded to obtain an
            embedding vector that will be used as query vector for the search operation.
            
            		- `processor`: A reference to an instance of an interface that
            provides preprocessing and encoding capabilities for images.
            		- `query_image`: The query image to be searched. Its type is inferred
            from the calling context, but it could be any valid Python object
            representing an image (e.g., a PIL Image or a BytesIO stream).
            		- `limit`: An optional parameter specifying the maximum number of
            results to return. Its type is `int`. If not provided, the default
            value is 2.
            		- `similarity_threshold`: An optional parameter specifying the minimum
            score threshold for eligible results. Its type is `float` between 0
            and 1, where 1 represents a strict similarity search and 0 represents
            an approximate search. If not provided, the default value is 0.5.
            
            	The rest of the function defines the search query and retrieves results
            from Qdrant, preprocesses and encodes the results, and displays them
            in an HTML table using Streamlit.
            
        limit (int): maximum number of search results to be retrieved from Qdrant,
            which is passed as an argument to the `search()` method of the Qdrant
            client object.
        similarity_threshold (float): threshold value for evaluating the similarity
            between the query image and the search results, where higher values
            lead to more selective retrieval of relevant images.

    """
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
    keys = ['variant_price', 'product_vendor', 'variant_title', 'product_title',  'product_handle', 'variant_featured_image']

    # Display the HTML in the Streamlit app
    st.markdown(generate_html_table(results, keys), unsafe_allow_html=True)

def main():
    # Set Streamlit page configuration
    """
    configures Streamlit page settings and displays a sidebar with controls for
    uploading an image and specifying search parameters. When the "Search" button
    is pressed, the function performs reverse image search using the uploaded image
    and user-provided limits for similar images and similarity threshold.

    """
    st.set_page_config(page_title="Manobo Market Intelligence")

    # Sidebar controls
    with st.sidebar:
        st.title("Settings")
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
        st.sidebar.image(query_image, caption='Uploaded Image', use_column_width=True)
        reverse_image_search(processor, query_image, limit=num_similar_images, similarity_threshold=similarity_threshold)

if __name__ == "__main__":
    main()
