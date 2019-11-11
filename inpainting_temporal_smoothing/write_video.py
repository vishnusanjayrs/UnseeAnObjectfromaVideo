import numpy as np
import cv2
import glob

img_array = []
# for filename in sorted(glob.glob('/home/vishnusanjay/cmpt820/deppfill_1/output/256_final/*rf*.png')):
#     print(filename)
#     img = cv2.imread(filename)
#     height, width, layers = img.shape
#     size = (width, height)
#     img_array.append(img)
for i in range(104,330):
    input_path = "/home/vishnusanjay/cmpt820/deppfill_1/input/hallway_" + str(i) + "_input1.png"
    inpainted_path = "/home/vishnusanjay/cmpt820/deppfill_1/output/own_video/output_" + str(i) + ".png"
    print(inpainted_path)
    l1_path = "/home/vishnusanjay/cmpt820/deppfill_1/output/own_video/rf_" + str(i) + ".png"
    # l2_path = "/home/vishnusanjay/cmpt820/deppfill_1/output/256_final/rf_" + str(i) + ".png"

    input_img = cv2.imread(input_path)
    inpaint_img = cv2.imread(inpainted_path)

    print(input_img.shape)
    print(inpaint_img.shape)

    l1_img = cv2.imread(l1_path)
    # l2_img = cv2.imread(l2_path)

    fin_img = np.concatenate((input_img,inpaint_img,l1_img),axis=1)
    # h2_img = np.concatenate((l1_img, l2_img),axis=1)
    # fin_img = np.concatenate((h_img,h2_img),axis=0)
    height, width,channels = fin_img.shape
    size = (width, height)
    img_array.append(fin_img)

print(len(img_array))

out = cv2.VideoWriter('/home/vishnusanjay/cmpt820/deppfill_1/own_video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

for i in range(len(img_array)):
    out.write(img_array[i])
    print(i)
out.release()