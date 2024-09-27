import image_service as ser
import image_segmentation as un
import numpy as np
import os

folder_path = "images/"
file_paths = file_names = os.listdir(folder_path)

#file_paths = ["street.png"]
gaus_values = [i / 10.0 for i in range(0, 10,2)]
k_values = [i*100 for i in range(1, 10,2)] + [1000,1500,3000,10000]

k_values = [500]
gaus_values = [0.9]

#rgb segmentation

for file_name in file_paths:
    for k in k_values:
        for gaus in gaus_values:
            file_path = folder_path+file_name
            pixel_matrix = ser.png_to_coloured_pixel_matrix(file_path,gaus)
            #print(pixel_matrix)
            rows, cols,_ = pixel_matrix.shape

            edge_list = ser.coloured_pixel_matrix_to_edge_list(pixel_matrix)
            edge_list.sort()

            segmentation = un.ImageSegmentation([((x,y),0) for x in range(rows) for y in range(cols)]
                                ,k=k)
            segmentation.run_segmentation(edge_list)
            seg_count = len(segmentation.get_groups())
            print(f"seg_count = {seg_count}")
            segmented_matrix = segmentation.get_image_matrix_with_components()
            if not os.path.exists("seg_"+file_name):
                os.makedirs("seg_"+file_name)
            # Example usage
            output_file_name = "seg_"+file_name+"/k="+str(k)+"_g="+str(gaus)+"_seg="+str(seg_count)+"_"+file_name
            output_file = output_file_name
            ser.pixel_matrix_to_image(segmented_matrix, output_file)



'''
#gray_scale segmentation
for file_name in file_paths:
    file_path = folder_path+file_name
    pixel_matrix = ser.gray_png_to_pixel_matrix(file_path,0.1)

    rows, cols = pixel_matrix.shape

    edge_list = ser.grey_pixel_matrix_to_edge_list(pixel_matrix)
    edge_list.sort()

    segmentation = un.ImageSegmentation([((x,y),0) for x in range(rows) for y in range(cols)]
                        ,k=1000)
    segmentation.run_segmentation(edge_list)
    segmented_matrix = segmentation.get_image_matrix_with_components()

    # Example usage
    output_file = f"segmented/seg_{file_name}"
    ser.pixel_matrix_to_image(segmented_matrix, output_file)

'''