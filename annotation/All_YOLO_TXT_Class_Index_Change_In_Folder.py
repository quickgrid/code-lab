## Change class index in txt file
## A crude implementation of yolo txt class index change

import glob
import numpy
import os


label_list = glob.glob("../dataset/labels/*.txt")
print(label_list)



for i in label_list:
  b = numpy.loadtxt(i, dtype=str, ndmin=2)



  img = cv2.imread("/content/dataset/images/" + os.path.basename(i).split(".")[0] + ".jpg")
  sz = img.shape 

  with open("../dataset/fixed/" + os.path.basename(i), 'w+') as target:
    num_rows = b.shape[0]

    output = []
    for j in range(0, num_rows):
      
      box = (float(b[j][1]), float(b[j][2]), float(b[j][3]), float(b[j][4]))
      x1, x2, x3, x4 = convertBox(sz, box)
      #print(x1, x2, x3, x4)


      ## Replace a certain class index in yolo with another class index
      if b[j][0] == "0":
        tmp = "1 " + str(x1) + " " + str(x2) + " " + str(x3) + " " + str(x4)
        
      #elif b[j][0] == "1":
      #  tmp = "1 " + str(x1) + " " + str(x2) + " " + str(x3) + " " + str(x4)
      
      #print(tmp)
      output.append(tmp)

    target.write('\n'.join(output))
