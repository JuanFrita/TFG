from PIL import Image

background = Image.open('D:/TFG/resources/Test_Training/test_pretrained/test/img_2.jpg')
foreground=  Image.open('D:/TFG/resources/Test_Training/test_pretrained/results/Figure_3.JPG')

if background.size != foreground.size:
    foreground = foreground.resize(background.size)
if background.mode != foreground.mode:
    foreground = foreground.convert(background.mode)

blended = Image.blend(background, foreground, alpha=0.6)
blended.show()