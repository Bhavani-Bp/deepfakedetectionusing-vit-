from PIL import Image
import numpy as np

# Create a random image (224x224 RGB)
img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
img = Image.fromarray(img_array)
img.save('test_image.jpg')
print("Created test_image.jpg")
