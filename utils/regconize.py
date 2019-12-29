import glob
import re
from .SamplePreprocessor import preprocess
import cv2
from .Model import Model, DecoderType
from .DataLoader import DataLoader, Batch

def infer(model, fnImg):
	print(fnImg)
	"recognize text in image provided by file path"
	img = preprocess(cv2.imread(fnImg, cv2.IMREAD_GRAYSCALE), Model.imgSize)
	batch = Batch(None, [img])
	(recognized, probability) = model.inferBatch(batch, True)
	# print('Recognized:', '"' + recognized[0] + '"')
	# print('Probability:', probability[0])
	return recognized[0]


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)',text)]

def regconizeText(img_path,model):
    links = glob.glob('{}/*.jpg'.format(img_path))
    links.sort(key=natural_keys)
    #links = open('/home/phucvinh98/LP_Project/HTR/data/test.txt').readlines()
    name = []
    accurate = 0

    for link in links:
        print(link)
        name.append(infer(model, link))
    return name