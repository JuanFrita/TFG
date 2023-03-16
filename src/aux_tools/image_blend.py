from PIL import Image

background = Image.open('C:/Users/juanf/OneDrive/Escritorio/TFG/resources/Test_Training/test_pretrained/test/img_4.jpg')
foreground=  Image.open('C:/Users/juanf/OneDrive/Escritorio/TFG/resources/Test_Training/test_pretrained/results/Figure_5.JPG')

if background.size != foreground.size:
    foreground = foreground.resize(background.size)
if background.mode != foreground.mode:
    foreground = foreground.convert(background.mode)

blended = Image.blend(background, foreground, alpha=0.6)
blended.show()