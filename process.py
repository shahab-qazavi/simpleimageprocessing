import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image, ImageChops
import imquality.brisque as br
import requests
from urllib.request import urlopen
from io import BytesIO

_url = 'https://media.wired.com/photos/5e59a85635982c0009f6eb8a/master/w_2560%2Cc_limit/python-popularity.jpg'


# def get_file_name(url):
#     name = ''
#     for item in url[::-1]:
#         if item != '/':
#             name += item
#         else:
#             break
#     return name[::-1]
#
#
# file_name = get_file_name(_url)
file = requests.get(url=_url, stream=True)
file_name = BytesIO(file.content)
cv2_file = urlopen(_url)
cv_img = np.asarray(bytearray(cv2_file.read()),dtype="uint8")
meta = cv2_file.info()

# img = cv2.imdecode(cv_img, cv2.IMREAD_COLOR)
# if file.status_code == 200:
#     with open(file_name, 'wb') as out_file:
#         shutil.copyfileobj(file.raw, out_file)


def visualize_colors(cluster, centroids):
    labels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
    (hist, _) = np.histogram(cluster.labels_, bins=labels)
    hist = hist.astype("float")
    hist /= hist.sum()

    colors = sorted([(percent, color) for (percent, color) in zip(hist, centroids)])
    colors_list = []
    count = 0
    for (percent, color) in colors:
        count += 1
        colors_list.append({f'c{count}': list(color), f'p{count}': "{:0.2f}%".format(percent * 100)})

    return colors_list


def check_image_has_border(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    # if bbox:
    #     return im.crop(bbox)
    return bbox != (0, 0, im.size[0], im.size[1])


image = cv2.imdecode(cv_img, cv2.IMREAD_COLOR)
height, width, _ = image.shape
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
reshape = image.reshape((image.shape[0] * image.shape[1], 3))

_cluster = KMeans(n_clusters=5).fit(reshape)
visualize = visualize_colors(_cluster, _cluster.cluster_centers_)
img = Image.open(file_name)
_format = img.format
quality = br.score(img)
# volume = os.stat(file_name).st_size
# volume = len(img.fp.read())
volume = meta.get(name="Content-Length")
check_border = check_image_has_border(img)

print({
    'Colors': visualize,
    'Format': _format,
    'Volume': f'{str(volume)} Bytes',
    'Quality': quality,
    'Height': height,
    'Width': width,
    'IsBorderRemoved': check_border
})
