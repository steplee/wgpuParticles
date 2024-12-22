import numpy as np, cv2, sys

for path in sys.argv[1:]:
    with open(path, 'rb') as fp:
        img = np.frombuffer(fp.read(), dtype=np.uint8)
        if img.size == 1920*1080*4:
            img = img.reshape(1080,1920,4)
        if img.size == 1280*720*4:
            img = img.reshape(720,1280,4)
        else:
            raise ValueError(f'unknown shape: {img.size}')
        cv2.imshow(path, img[...,[2,1,0]])
cv2.waitKey(0)
