import matplotlib.pyplot as plt
import matplotlib.image as mpimg

path = "/home/natalie/train/synthetic/scenenet/train/0/0/instance/0.png"

img = mpimg.imread(path)
plt.imshow(img)
plt.show()
