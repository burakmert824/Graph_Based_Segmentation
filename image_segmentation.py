import numpy as np
import random
'''
This class provides functionality for segmenting an image into regions based on a graph-based segmentation algorithm.
'''
class ImageSegmentation:

    '''
    Initializes the ImageSegmentation object.
    '''
    def __init__(self, nodes, k):
        self.parent = {}
        self.rank = {}
        self.size = {}
        self.max_edge = {}
        self.image_size = (0,0)
        self.k = k
        
        for node, value in nodes:
            self.parent[node] = node
            self.rank[node] = 0
            self.size[node] = 1
            self.max_edge[node] = value
            self.image_size = max(self.image_size,node)
    '''
    Computes the threshold function for a given component size.
    '''
    def t_f(self, size_c):
        return self.k / size_c
    '''
    Finds the root of the component containing node x using path compression.
    '''
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    '''
    Computes the minimum internal difference between two components.
    '''
    def Mint(self, root_x, root_y):
        size_x = self.size[root_x]
        size_y = self.size[root_y]
        return min(self.t_f(size_x) + self.max_edge[root_x],
                   self.t_f(size_y) + self.max_edge[root_y])
    '''
      Performs the union operation to merge two components if the edge satisfies the minimum internal difference condition.
    '''
    def union(self,edge_value, x, y):
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return
        
        if self.Mint(root_x, root_y) < edge_value:
            return
        
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
            self.size[root_y] += self.size[root_x]
            self.max_edge[root_y] = edge_value
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
            self.size[root_x] += self.size[root_y]
            self.max_edge[root_x] = edge_value
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
            self.size[root_x] += self.size[root_y]
            self.max_edge[root_x] = edge_value
    '''
      Retrieves the groups of nodes after segmentation.
    '''
    def get_groups(self):
        groups = {}
        for node in self.parent:
            root = self.find(node)
            if root not in groups:
                groups[root] = []
            groups[root].append(node)
        return list(groups.values())
    

    '''
          Generates an image matrix with each pixel colored according to its component.
    '''
    def get_image_matrix_with_components(self):
        # Find the dimensions of the original image
        max_x = self.image_size[1] + 1
        max_y = self.image_size[0] + 1

        # Generate random (r, g, b) tuples for each unique root
        root_to_color = {}
        unique_roots = set(self.parent.values())
        #print(unique_roots)
        unique_colors = set()
        for root in unique_roots:
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            root_to_color[root] = (r, g, b)

        # Create a matrix to store pixels and their component roots
        new_matrix = np.zeros((max_y, max_x, 3), dtype=int)

        # Populate the matrix with pixels and their component root colors
        for node, root in self.parent.items():
            new_matrix[node[0], node[1]] = root_to_color[root]
        return new_matrix
    
    '''
      Runs the segmentation algorithm on the sorted list of edges.
    '''
    def run_segmentation(self,sorted_edges):
        for edge in sorted_edges:
            self.union(edge[0],edge[1],edge[2])
        