from PIL import Image

background = Image.open('D:/TFG/resources/Test_Training/test_trained/test/img_0.jpg')
foreground=  Image.open('D:/TFG/resources/Test_Training/test_trained/results/img_0_result2.JPG')

if background.size != foreground.size:
    foreground = foreground.resize(background.size)
if background.mode != foreground.mode:
    foreground = foreground.convert(background.mode)

blended = Image.blend(background, foreground, alpha=0.6)
blended.show()