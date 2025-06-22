import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import io
import os

# Configure the page
st.set_page_config(
    page_title="CNN Image Classifier",
    page_icon="🖼️",
    layout="wide"
)

# Title and description
st.title("🖼️ CNN Image Classification")
st.markdown("Upload an image to get predictions from your trained CNN model.")

# Sidebar for model configuration
st.sidebar.header("Model Configuration")

# Model and test_df upload
uploaded_model = st.sidebar.file_uploader(
    "Upload your CNN model (.h5 file)",
    type=['h5'],
    help="Upload your trained CNN model file"
)

uploaded_test_df = st.sidebar.file_uploader(
    "Upload test_df (.pkl file)",
    type=['pkl'],
    help="Upload the test_df pickle file for label mapping"
)

# Image preprocessing settings
img_size = (299, 299)  # Fixed for Xception
st.sidebar.subheader("Model Information")
st.sidebar.write(f"**Input size:** {img_size}")
st.sidebar.write("**Preprocessing:** Xception preprocess_input")

# Load model and test_df
@st.cache_resource
def load_cnn_model(model_file):
    """Load the CNN model from uploaded file"""
    try:
        # Save uploaded file temporarily
        model_bytes = model_file.read()
        with open("temp_model.h5", "wb") as f:
            f.write(model_bytes)
        
        # Load model
        model = tf.keras.models.load_model("temp_model.h5")
        return model, None
    except Exception as e:
        return None, str(e)

@st.cache_data
def load_test_dataframe(df_file):
    """Load the test dataframe from uploaded pickle file"""
    try:
        # Save uploaded file temporarily
        df_bytes = df_file.read()
        with open("temp_test_df.pkl", "wb") as f:
            f.write(df_bytes)
        
        # Load dataframe
        test_df = pd.read_pickle("temp_test_df.pkl")
        
        # Create label mapping
        pairs = test_df[['label', 'label_name']].drop_duplicates(subset='label')
        label_map = pairs.set_index('label')['label_name'].to_dict()
        
        return test_df, label_map, None
    except Exception as e:
        return None, None, str(e)

def get_img_array_from_pil(pil_image, size):
    """Convert PIL image to preprocessed array"""
    # Resize image to target size
    img = pil_image.resize(size)
    
    # Convert PIL image to array
    array = keras.utils.img_to_array(img)
    
    # Add batch dimension
    array = np.expand_dims(array, axis=0)
    
    return array

def class_prediction(img_array, model, label_map):
    """Make prediction on preprocessed image array"""
    try:
        # Apply Xception preprocessing
        preprocess_input = keras.applications.xception.preprocess_input
        preprocessed_array = preprocess_input(img_array)
        
        # Get predictions
        preds = model.predict(preprocessed_array)
        probs = preds[0]
        
        # Get top prediction
        top_index = np.argmax(probs)
        top_confidence = probs[top_index]
        
        # Get predicted class name
        predicted_class_name = label_map.get(top_index, f"Unknown Class {top_index}")
        
        return predicted_class_name, top_confidence, probs, top_index, None
    
    except Exception as e:
        return None, None, None, None, str(e)

# Main app
if uploaded_model and uploaded_test_df:
    # Load model and test_df
    with st.spinner("Loading model and test data..."):
        model, model_error = load_cnn_model(uploaded_model)
        test_df, label_map, df_error = load_test_dataframe(uploaded_test_df)
    
    if model is None:
        st.error(f"Error loading model: {model_error}")
    elif test_df is None:
        st.error(f"Error loading test_df: {df_error}")
    else:
        st.success("✅ Model and test data loaded successfully!")
        
        # Display model and data info
        st.subheader("Model & Data Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write(f"**Input shape:** {model.input_shape}")
            st.write(f"**Output shape:** {model.output_shape}")
        
        with col2:
            st.write(f"**Total parameters:** {model.count_params():,}")
            st.write(f"**Number of layers:** {len(model.layers)}")
        
        with col3:
            st.write(f"**Number of classes:** {len(label_map)}")
            st.write(f"**Test samples:** {len(test_df)}")
        
        # Display available classes
        with st.expander("View Available Classes"):
            sorted_classes = sorted(label_map.items())
            for label_idx, class_name in sorted_classes:
                st.write(f"**{label_idx}:** {class_name}")
        
        # Image upload section
        st.subheader("Upload Image for Classification")
        
        uploaded_image = st.file_uploader(
            "Choose an image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image file for classification"
        )
        
        if uploaded_image is not None:
            # Display uploaded image
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Uploaded Image")
                image = Image.open(uploaded_image)
                st.image(image, caption="Original Image", use_column_width=True)
                
                # Show image info
                st.write(f"**Original size:** {image.size}")
                st.write(f"**Image mode:** {image.mode}")
                st.write(f"**Will be resized to:** {img_size}")
            
            with col2:
                st.subheader("Prediction Results")
                
                # Convert to RGB if needed
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Preprocess and predict
                with st.spinner("Making prediction..."):
                    img_array = get_img_array_from_pil(image, img_size)
                    predicted_class, confidence, all_probs, class_index, error = class_prediction(
                        img_array, model, label_map
                    )
                
                if predicted_class is not None:
                    # Display main prediction
                    st.success(f"**Predicted Class:** {predicted_class}")
                    st.info(f"**Confidence:** {confidence:.3f} ({confidence:.1%})")
                    st.write(f"**Class Index:** {class_index}")
                    
                    # Display top 5 predictions
                    st.subheader("Top 5 Predictions")
                    
                    # Get top 5 indices and probabilities
                    top_indices = np.argsort(all_probs)[-5:][::-1]
                    
                    for i, idx in enumerate(top_indices):
                        class_name = label_map.get(idx, f"Unknown Class {idx}")
                        prob = all_probs[idx]
                        
                        # Highlight the top prediction
                        if i == 0:
                            st.markdown(f"🥇 **{class_name}**: {prob:.3f} ({prob:.1%})")
                        elif i == 1:
                            st.markdown(f"🥈 **{class_name}**: {prob:.3f} ({prob:.1%})")
                        elif i == 2:
                            st.markdown(f"🥉 **{class_name}**: {prob:.3f} ({prob:.1%})")
                        else:
                            st.write(f"{i+1}. **{class_name}**: {prob:.3f} ({prob:.1%})")
                        
                        st.progress(float(prob))
                
                else:
                    st.error(f"Error making prediction: {error}")
        
        # Test with random sample from test_df
        st.subheader("🎲 Test with Random Sample")
        if st.button("Get Random Test Sample Info"):
            random_sample = test_df.sample(1).iloc[0]
            st.write(f"**Sample Label:** {random_sample['label']}")
            st.write(f"**Sample Label Name:** {random_sample['label_name']}")
            if 'image_path' in random_sample:
                st.write(f"**Image Path:** {random_sample['image_path']}")
        
        # Additional information
        st.subheader("💡 Model Details")
        st.markdown(f"""
        - **Preprocessing**: Uses Xception's preprocess_input function
        - **Input Size**: Images are resized to {img_size[0]}x{img_size[1]} pixels
        - **Image Format**: Automatically converts to RGB if needed
        - **Batch Processing**: Single image prediction with batch dimension
        - **Output**: Probability distribution over {len(label_map)} classes
        """)

elif uploaded_model and not uploaded_test_df:
    st.warning("⚠️ Please upload both the model file (.h5) and test_df file (.pkl) to proceed.")
    st.info("The test_df is required to map class indices to readable class names.")
    
elif uploaded_test_df and not uploaded_model:
    st.warning("⚠️ Please upload both the model file (.h5) and test_df file (.pkl) to proceed.")
    st.info("Both files are required for the application to work properly.")

else:
    # Instructions when no files are uploaded
    st.info("👆 Please upload both your CNN model file (.h5) and test_df file (.pkl) in the sidebar to get started.")
    
    st.subheader("How to use this app:")
    st.markdown("""
    1. **Upload your model**: Use the sidebar to upload your trained CNN model (.h5 file)
    2. **Upload test_df**: Upload the test_df.pkl file for label mapping
    3. **Upload an image**: Choose an image file to classify
    4. **View results**: See the prediction results with class names and confidence scores
    """)
    
    st.subheader("Required files:")
    st.markdown("""
    - **Model file**: .h5 format (TensorFlow/Keras model)
    - **Test dataframe**: .pkl format (contains label to label_name mapping)
    - **Image files**: PNG, JPG, JPEG formats supported
    """)
    
    st.subheader("Expected file structure:")
    st.code("""
    test_df columns should include:
    - 'label': numerical class indices
    - 'label_name': readable class names
    """)