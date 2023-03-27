from PIL import Image

background = Image.open('D:/TFG/resources/Test_Training/test_trained/test/img_5.jpg')
foreground=  Image.open('D:/TFG/resources/Test_Training/test_trained/results/img_5_result.JPG')

if background.size != foreground.size:
    background = background.resize(foreground.size)
if background.mode != foreground.mode:
    foreground = foreground.convert(background.mode)

blended = Image.blend(background, foreground, alpha=0.6)
blended.show()