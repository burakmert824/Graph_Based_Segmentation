from PIL import Image
import numpy as np
import math
from scipy.ndimage import convolve
from scipy.ndimage import gaussian_filter

"""
    Apply Gaussian smoothing to the input image.
"""
def gaussian_smoothing(image, sigma):
    smoothed_image = gaussian_filter(image, sigma)
    return smoothed_image
"""
    Converts a grayscale PNG image to a matrix of pixel values.
"""    
def gray_png_to_pixel_matrix(file_path,smoothing_constant = 0.8):
    # Open the PNG image file
    image = Image.open(file_path)
    # Convert the image to grayscale
    image = image.convert("L")
    # Convert the image to a numpy array
    pixel_matrix = np.array(image)
    pixel_matrix = gaussian_smoothing(pixel_matrix,smoothing_constant)
    print("Gray Scale image:",file_path,"'s shape :",pixel_matrix.shape)
    return pixel_matrix

"""
    This function converts a PNG image file to a colored pixel matrix, applying Gaussian smoothing to the pixel values.
"""
def png_to_coloured_pixel_matrix(file_path,smoothing_constant = 0.8):
    # Open the PNG image file
    image = Image.open(file_path)
    # Convert the image to a numpy array
    pixel_matrix = np.array(image)
    # Check if the array has an alpha channel (4th dimension)
    if pixel_matrix.shape[2] > 3:
        # Remove the alpha channel by selecting only the RGB channels
        pixel_matrix = pixel_matrix[:, :, :3]
    pixel_matrix = gaussian_smoothing(pixel_matrix, smoothing_constant)
    
    print("Coloured", file_path, "'s shape:", pixel_matrix.shape)
    return pixel_matrix
"""
    This function converts a PNG image file to separate red, green, and blue matrices, applying Gaussian smoothing to each channel separately.
"""
def png_to_rgb_matrices(file_path,smoothing_constant = 0.8):
    # Open the PNG image file
    image = Image.open(file_path)
    # Convert the image to a numpy array
    pixel_matrix = np.array(image)
    
    # Separate the image array into its red, green, and blue channels
    red_channel = pixel_matrix[:, :, 0]
    green_channel = pixel_matrix[:, :, 1]
    blue_channel = pixel_matrix[:, :, 2]

    # Apply Gaussian smoothing to each channel separately
    red = gaussian_smoothing(red_channel, smoothing_constant)
    green = gaussian_smoothing(green_channel, smoothing_constant)
    blue = gaussian_smoothing(blue_channel, smoothing_constant)

    print("Coloured", file_path, "'s shape:", pixel_matrix.shape)
    return red,green,blue
"""
    Converts a pixel matrix to an image and saves it to a file.
"""
def pixel_matrix_to_image(pixel_matrix, output_file):
    pixel_matrix = pixel_matrix.astype(np.uint8)
    # Convert the pixel matrix to an image
    image = Image.fromarray(pixel_matrix)
    # Save the image to a file
    image.save(output_file)

"""
    Finds neighboring pixels of a given pixel in a 2D grid.
"""
def find_neighbors_8_way(row, col, rows, cols):
    
    neighbors = []
    for i in range(max(0, row - 1), min(rows, row + 2)):
        for j in range(max(0, col - 1), min(cols, col + 2)):
            if i != row or j != col:
                neighbors.append((i, j))
    return neighbors


"""
    Converts a pixel matrix into a graph represented as an edge list.
"""
def grey_pixel_matrix_to_edge_list(pixel_matrix):

    rows, cols = pixel_matrix.shape
    edge_list = []
    
    # Iterate over each pixel
    for row in range(rows):
        for col in range(cols):
            # Get pixel value
            pixel_value = pixel_matrix[row, col]

            # Get neighboring pixels
#            neighbors = find_neighbors_8_way(row, col, rows, cols)
            neighbors = find_neighbors_8_way(row, col, rows, cols)

            # Iterate over neighboring pixels
            for neighbor_row, neighbor_col in neighbors:
                # Get neighbor pixel value
                neighbor_pixel_value = pixel_matrix[neighbor_row, neighbor_col]

                # Calculate edge weight (absolute difference)
                edge_weight = abs(int(pixel_value) - int(neighbor_pixel_value))

                # Add edge to edge list
                edge_list.append((edge_weight,(row, col), (neighbor_row, neighbor_col)))

    return edge_list


"""
    Converts a pixel matrix into a graph represented as an edge list.
"""
def coloured_pixel_matrix_to_edge_list(pixel_matrix):
    def vector_distance(point1, point2):
        """
            Calculate the Euclidean distance between two points in three-dimensional space.
        """
        #print(point1)
        # Unpack the coordinates of the points
        [x1, y1, z1] = point1.astype(np.int64)
        [x2, y2, z2] = point2.astype(np.int64)
        p = point1
        t = point2
        #print(point1)
        # Calculate the differences in coordinates
        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1
        

        # Calculate the squared sum of differences
        squared_distance = dx**2 + dy**2 + dz**2

        # Calculate the square root of the squared distance to get the Euclidean distance
        distance = math.sqrt(squared_distance)

        return distance
    rows, cols, _ = pixel_matrix.shape
    edge_list = []
    pixel_matrix = pixel_matrix.astype(np.int64)
    # Iterate over each pixel
    for row in range(rows):
        for col in range(cols):
            # Get pixel value
            pixel_value = pixel_matrix[row, col]

            # Get neighboring pixels
            neighbors = find_neighbors_8_way(row, col, rows, cols)

            # Iterate over neighboring pixels
            for neighbor_row, neighbor_col in neighbors:
                # Get neighbor pixel value
                neighbor_pixel_value = pixel_matrix[neighbor_row, neighbor_col]

                # Calculate edge weight (absolute difference)
                edge_weight = vector_distance(pixel_value,neighbor_pixel_value)

                # Add edge to edge list
                edge_list.append((edge_weight,(row, col), (neighbor_row, neighbor_col)))

    return edge_list
'''
# Example usage
file_path = "obj10__25.png"
pixel_matrix = gray_png_to_pixel_matrix(file_path)
print(pixel_matrix)

print(pixel_matrix.shape)

edge_list = grey_pixel_matrix_to_edge_list(pixel_matrix)
print(edge_list[:10])


# Example usage
output_file = "output_image1.png"
pixel_matrix_to_image(pixel_matrix, output_file)
'''