# 福州大学-洪博杨 无人机检测项目

## yolov7概述

### 1. YOLOV7 整体结构



![e69186f0cbe46371a5aedbb20d6b6f8](.\img\e69186f0cbe46371a5aedbb20d6b6f8.png)



我们先整体来看下 YOLOV7，首先对输入的图片 resize 为 640x640 大小，输入到 backbone 网络中，然后经 head 层网络输出三层不同 size 大小的 **feature map**，经过 Rep 和 conv输出预测结果，这里以 coco 为例子，输出为 80 个类别，然后每个输出(x ,y, w, h, o) 即坐标位置和前后背景，3 是指的 anchor 数量，因此每一层的输出为 (80+5)x3 = 255再乘上 feature map 的大小就是最终的输出了。



### 2. backbone

YOLOV7 的 backbone 如下图所示

![img](https://pic2.zhimg.com/80/v2-0ef5b6837f287c20c1e03074ec60f8c1_720w.webp)

总共有 50 层, 我在上图用黑色数字把关键层数标示出来了。
首先是经过 **4** 层卷积层，如下图，**CBS** 主要是 **Conv + BN + SiLU** 构成，我在图中用不同的颜色表示不同的 size 和 stride, 如 (3, 2) 表示卷积核大小为 3 ，步长为 2。 在 config 中的配置如图。



![img](https://pic4.zhimg.com/80/v2-f5bfcc492abb9d38d6ffab0ed05a0637_720w.webp)

![img](https://pic2.zhimg.com/80/v2-8da343d878bfe713624acb8bac87c4fd_720w.webp)





经过 4个 CBS 后，特征图变为 160 * 160 * 128 大小。随后会经过论文中提出的 ELAN 模块，ELAN 由多个 CBS 构成，其输入输出特征大小保持不变，通道数在开始的两个 CBS 会有变化， 后面的几个输入通道都是和输出通道保持一致的，经过最后一个 CBS 输出为需要的通道。



![img](https://pic4.zhimg.com/80/v2-51e6aa08680addc9ca70f1b2048a6c27_720w.webp)



![img](https://pic4.zhimg.com/80/v2-7a61c958f4d64209a5bc94ba59687803_720w.webp)

![img](https://pic2.zhimg.com/80/v2-b5f33f8c851bda6c245921e1cc70f629_720w.webp)

**MP** 层 主要是分为 Maxpool 和 CBS , 其中 MP1 和 MP2 主要是通道数的比变化。

![img](https://pic4.zhimg.com/80/v2-75a8f60c85ce28b69c528af09be1483f_720w.webp)

backbone的基本组件就介绍完了，我们整体来看下 backbone，经过 4 个 CBS 后，接入例如一个 ELAN ，然后后面就是三个 MP + ELAN 的输出，对应的就是 C3/C4/C5 的输出，大小分别为 80 * 80 * 512 ， 40 * 40 * 1024， 20 * 20 * 1024。 每一个 MP 由 5 层， ELAN 有 8 层， 所以整个 backbone 的层数为 4 + 8 + 13 * 3 = 51 层， 从 0 开始的话，最后一层就是第 50 层。

### 3. head



![img](https://pic3.zhimg.com/80/v2-f9abc72be912e63bd51771030867532e_720w.webp)

![img](https://pic2.zhimg.com/80/v2-dd65194515f4637117ecba3ae20a49d1_720w.webp)

![img](https://pic2.zhimg.com/80/v2-3b8b49de2970b255217b848c146bbbdd_720w.webp)





YOLOV7 head 其实就是一个 **pafpn** 的结构，和之前的YOLOV4，YOLOV5 一样。首先，对于 backbone 最后输出的 32 倍降采样特征图 C5，然后经过 SPPCSP，通道数从1024变为512。先按照 top down 和 C4、C3融合，得到 P3、P4 和 P5；再按 bottom-up 去和 P4、P5 做融合。这里基本和 YOLOV5 是一样的，区别在于将 YOLOV5 中的 CSP 模块换成了 ELAN-H 模块， 同时下采样变为了 MP2 层。

ELAN-H 模块是我自己命名的，它和 backbone 中的 ELAN 稍微有点区别就是 cat 的数量不同。



![img](https://pic3.zhimg.com/80/v2-7a18558c9d458980b332eb32d680898a_720w.webp)



至于 E-ELAN 论文中说到是在 YOLOV7-E6eE中用到, 但是我在看代码的时候，有点没太明白是怎么做 shuffle 的，这里等我弄明白了再补上来。



![img](https://pic1.zhimg.com/80/v2-99813d788bd2b050e054225a68e2f028_720w.webp)



对于 **pafpn** 输出的 P3、P4 和 P5 ， 经过 RepConv 调整通道数，最后使用 1x1 卷积去预测 objectness、class 和 bbox 三部分。

RepConv 在训练和推理是有一定的区别。训练时有三个分支的相加输出，部署时会将分支的参数重参数化到主分支上。

![img](https://pic4.zhimg.com/80/v2-1f39f33049b838cde8996c08343e1673_720w.webp)

## 训练与预测过程

### 模型标签转换

原标签是自定义格式的标签，需要将自定义格式的标签转化为yolo格式或者voc格式或者COCO格式的标签

以下是转化为COCO格式标签的代码

``` python
import json
import os
root = '/home/aistudio/wurenji/train/'
dir = os.listdir(root)
from sahi.utils.coco import Coco, CocoCategory, CocoImage, CocoAnnotation
from sahi.utils.file import save_json
from PIL import Image

coco = Coco()
coco.add_category(CocoCategory(id=0, name='uav'))
for folder in dir:
  print(folder)
  with open(root+folder+'/IR_label.json','r',encoding='utf8') as fp1:
      json_data = json.load(fp1)
      num = len(os.listdir(root+folder))
      print(num)
      
      for i in range(1,num-1):
        if i%10!=3:
          os.remove(root+folder+"/"+str(i).zfill(6)+".jpg")
          continue
        imgname = root+folder+"/"+str(i).zfill(6)+".jpg"
        width, height = Image.open(imgname).size
        coco_image = CocoImage(file_name=imgname, height=height, width=width)
        ann = json_data["gt_rect"][i-1]
        if(json_data['exist'][i-1]==0):
          continue
        # print(root+folder+str(i))
        coco_image.add_annotation(
            
            CocoAnnotation(
                bbox=[ann[0], ann[1], ann[2], ann[3]],
                category_id=0,
                category_name='uav'
            )
        )
        coco.add_image(coco_image)
!mkdir ./wurenji/annotations
save_json(data=coco.json, save_path='./wurenji/annotations/test.json')
```

由于本次项目没有实体的GPU支持，于是需要使用云服务器的GPU进行训练，或使用COLAB服务或者百度的AISTUDIO服务

本项目使用百度的AISTUDIO服务进行训练与预测

### 训练代码

``` shel
git clone https://gitee.com/monkeycc/PaddleYOLO.git # clone
cd PaddleYOLO
pip install -r PaddleYOLO/requirements.txt  # install

python -m paddle.distributed.launch --selected_gpus="0" ./PaddleYOLO/tools/train.py -c ./PaddleYOLO/configs/yolov7/yolov7_tiny_300e_coco.yml --eval --amp 
```

### 预测代码

``` python
import os
import json
folders = os.listdir("/home/aistudio/wurenji/test")
for folder in folders:
    os.system("python /home/aistudio/PaddleYOLO/tools/infer.py -c /home/aistudio/PaddleYOLO/configs/yolov7/yolov7_tiny_300e_coco.yml -o weights=/home/aistudio/output/yolov7_tiny_300e_coco/model_final.pdparams --infer_dir=/home/aistudio/wurenji/test/"+folder+" --save_results ./test --visualize False")
    bbox = "/home/aistudio/output/bbox"
    bbox_data=[]
    with open("/home/aistudio/output/bbox.json",'r',encoding='utf8') as fp1:
        bbox_data = json.load(fp1)
    os.system("mv /home/aistudio/output/bbox.json /home/aistudio/ans/"+folder+".json")
    os.system("rm -rf /home/aistudio/output/bbox.json")
    os.system("rm -rf /home/aistudio/output/*.jpg")
```

### 样例展示

![000541](.\img\000541.jpg)