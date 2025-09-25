import os
from PIL import Image

def resize_images(input_folder, output_folder, width=320, height=256):
    # Make sure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)

        # Only process image files
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            try:
                # Open image
                img = Image.open(file_path)
                # Resize (width, height)
                img_resized = img.resize((width, height), Image.BILINEAR)

                # Save to output folder
                output_path = os.path.join(output_folder, filename)
                img_resized.save(output_path)

                print(f"Saved resized image: {output_path}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Example usage:
input_folder = "./examples/scared_9_1/"
output_folder = "./examples/scared_9_1_/"
resize_images(input_folder, output_folder, width=320, height=256)