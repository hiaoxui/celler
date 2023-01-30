import imagej
import scyjava
scyjava.config.add_option('-Xmx24g')
ij = imagej.init("2.5.0", mode='interactive')
print(ij.getVersion())
img_path = '/home/hiaoxui/images/Ax2 mry-C2B-iai_AcquisitionBlock3.czi/Ax2 mry-C2B-iai_AcquisitionBlock3_pt3.tif'
dataset = ij.io().open(img_path)
ij.ui().show(dataset)
x = 1
