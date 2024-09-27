import numpy as np
from sklearn.cluster import KMeans
from PIL import Image

# Function to load and preprocess the image
def load_image(image_path):
    # Open the image using PIL
    image = Image.open(image_path)
    # Convert the image to an RGB format if it's not
    image = image.convert('RGB')
    # Convert the image to a NumPy array
    image_np = np.array(image)
    return image_np

# Function to perform KMeans clustering for image quantization
def image_compression(image_np, n_colors):
    # Get the dimensions of the image
    height, width, channels = image_np.shape
    
    # Reshape the image array to be a 2D array of pixels
    image_2d = image_np.reshape(-1, 3)  # shape (height*width, 3)
    
    # Apply KMeans clustering to reduce the number of colors
    kmeans = KMeans(n_clusters=n_colors, random_state=0)
    kmeans.fit(image_2d)
    
    # Get the cluster centers (which are the colors) and labels for each pixel
    colors = kmeans.cluster_centers_.astype(np.uint8)  # Convert to uint8 to match image format
    labels = kmeans.labels_
    
    # Reconstruct the image with the quantized colors
    quantized_image_2d = colors[labels]
    quantized_image_np = quantized_image_2d.reshape(height, width, channels)
    
    return quantized_image_np.astype(np.uint8) 

# Function to concatenate and save the original and quantized images side by side
def save_result(original_image_np, quantized_image_np, output_path):
    # Convert NumPy arrays back to PIL images
    original_image = Image.fromarray(original_image_np)
    quantized_image = Image.fromarray(quantized_image_np)
    
    # Get dimensions
    width, height = original_image.size
    
    # Create a new image that will hold both the original and quantized images side by side
    combined_image = Image.new('RGB', (width * 2, height))
    
    # Paste original and quantized images side by side
    combined_image.paste(original_image, (0, 0))
    combined_image.paste(quantized_image, (width, 0))
    
    # Save the combined image
    combined_image.save(output_path)

def __main__():
    # Load and process the image
    image_path = 'favorite_image.png'  
    output_path = 'compressed_image.png'  
    image_np = load_image(image_path)

    # Perform image quantization using KMeans
    n_colors = 8  # Number of colors to reduce the image to, you may change this to experiment
    quantized_image_np = image_compression(image_np, n_colors)

    # Save the original and quantized images side by side
    save_result(image_np, quantized_image_np, output_path)

# Uncomment to run the script
__main__()
