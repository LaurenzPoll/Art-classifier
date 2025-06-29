import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from keras import Model
import matplotlib as mpl

# Configure the page
st.set_page_config(
    page_title="CNN Image Classifier",
    page_icon="🖼️",
    layout="wide"
)

# Load model and test_df
@st.cache_resource
def load_cnn_model(model_path):
    """Load the CNN model"""
    try:
        model = tf.keras.models.load_model(model_path)

        return model, None
    except Exception as e:
        return None, str(e)

@st.cache_data
def load_test_dataframe(df_path):
    """Load the test dataframe from uploaded pickle file"""
    try:
        test_df = pd.read_pickle(df_path)
        
        pairs = test_df[['label', 'label_name']].drop_duplicates(subset='label')
        label_map = pairs.set_index('label')['label_name'].to_dict()
        
        return test_df, label_map, None
    except Exception as e:
        return None, None, str(e)

# Title and description
st.title("🖼️ CNN Image Classification")
st.markdown("Upload an image to get predictions from the trained CNN model")

# Sidebar for model configuration
st.sidebar.header("Model and Data")
MODEL_PATH = "../model/iteration_0/cnn_model.h5"
TEST_DF_PATH = "../model/iteration_0/test_df.pkl"

with st.spinner("Loading model and test data..."):
    model, error = load_cnn_model(MODEL_PATH)
    if error:
        st.sidebar.write(f"Failed loading model from `{MODEL_PATH}`")
    else:
        st.sidebar.success("Model loaded successfully!")

    test_df, label_map, error = load_test_dataframe(TEST_DF_PATH)
    if error:
        st.sidebar.write(f"Failed loading test data from `{TEST_DF_PATH}`")
    else:
        st.sidebar.success("Test data loaded successfully!")


st.sidebar.write(f"**Input shape:** {model.input_shape}")
st.sidebar.write(f"**Output shape:** {model.output_shape}")
st.sidebar.write(f"**Total parameters:** {model.count_params():,}")
st.sidebar.write(f"**Number of layers:** {len(model.layers)}")
st.sidebar.write(f"**Number of classes:** {len(label_map)}")
st.sidebar.write(f"**Test samples:** {len(test_df)}")


with st.sidebar.expander("View Available Classes"):
    sorted_classes = sorted(label_map.items())
    for label_idx, class_name in sorted_classes:
        st.write(f"**{label_idx}:** {class_name}")


# Image preprocessing settings
img_size = (299, 299)  # Fixed for Xception
st.sidebar.subheader("Model Information")
st.sidebar.write(f"**Input size:** {img_size}")
st.sidebar.write("**Preprocessing:** Xception preprocess_input")


pooling_layer = model.get_layer("global_average_pooling2d_2")
#    └── its input tensor is the 10×10×2048 conv-feature-map you want.

# 2) Build a new “grad-model” that returns both
#      a) that conv-map, and
#      b) your final predictions
grad_model = Model(
    inputs = model.inputs,
    outputs = [ pooling_layer.input, model.output ]
)

preprocess_input = keras.applications.xception.preprocess_input

def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.utils.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.utils.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array

def get_img_array_from_pil(pil_image, size):
    """Convert PIL image to preprocessed array"""
    # Resize image to target size
    img = pil_image.resize(size)
    
    # Convert PIL image to array
    array = keras.utils.img_to_array(img)
    
    # Add batch dimension
    array = np.expand_dims(array, axis=0)
    
    return array

def make_gradcam_heatmap(img_array, grad_model, pred_index=None):
    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam(img_path, heatmap, alpha=0.4):
    # Load the original image
    img = keras.utils.load_img(img_path)
    img = keras.utils.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = mpl.colormaps["jet"]

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)

    return superimposed_img

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
if not error:
    # Image upload section
    st.subheader("Upload Image for Classification")
    
    uploaded_image = st.file_uploader(
        "Choose an image...",
        type=['png', 'jpg', 'jpeg'],
        help="Upload an artwork for classification"
    )
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image)

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        img_array = get_img_array_from_pil(image, img_size)
            
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Uploaded Image")
            # image = Image.open(uploaded_image)
            st.image(image, caption="Original Image", use_column_width=True)
            
            # Show image info
            st.write(f"**Original size:** {image.size}")
            st.write(f"**Image mode:** {image.mode}")
            st.write(f"**Will be resized to:** {img_size}")
        
        with col2:
            st.subheader("Prediction Results")

            # Preprocess and predict
            with st.spinner("Making prediction..."):
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
                st.subheader("Grad-CAM")
                img_array = preprocess_input(get_img_array(uploaded_image, size=img_size))
                heatmap = make_gradcam_heatmap(img_array, grad_model)
                gradcam_img = display_gradcam(uploaded_image, heatmap, alpha=0.4)
                st.image(gradcam_img, caption="Grad-CAM Overlay", use_column_width=True)
            
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
else:
    # Instructions when no files are uploaded
    st.info("👆 Please make sure both your CNN model file (.h5) and test_df file (.pkl) are available.")
    
    st.subheader("How to use this app:")
    st.markdown("""
    1. **Upload an image**: Choose an image file to classify
    2. **View results**: See the prediction results with class names and confidence scores
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