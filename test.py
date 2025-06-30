
import cv2
import urllib.request

import numpy as np
d =urllib.request.urlretrieve('http://127.0.0.1:8090/api/files/collection/xd1mivsg192584q/s4zmi7yhrex_99mlpz1l83.unknown_1.jpg', "uploads/local-filename.jpg")

print(d[0])