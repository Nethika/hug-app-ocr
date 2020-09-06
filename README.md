# Credit Card Reader

## Techniques Used:

- CTPN
- tesseract
- fast RCNN
- reference method with OpenCV contours.
- doc scanner

## To Run:

```python read_credit_card.py --image images/credit_card_01.png```

```python read_credit_card.py --image images/images47.jpg```


## Hug Version

```hug -f run_hug.py``


## Output Images

Preprossesed image: ```processed_output/```

draw boxes from ctpn: ```data/results/```

identified info : ```data/text_read/```

## To Cython to work:

Change line #4 in `lib/utils/make.sh` as appropriate with `python` or `python3`

compile:

```
cd utils/bbox
chmod +x make.sh
./make.sh
```

## To work with Tensorflow 2:

lines 330,331 of `read_credit_card.py`

```    
# init session
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
sess = tf.compat.v1.Session(config=config)
```

or:
```
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
```


## Requirements (in docker container):

```
pip3 install keras
pip3 install imutils
pip3 install matplotlib==2.0.2
'tkinter': apt-get install python3-tk
pip3 install opencv-python==3.4.3.18
```


