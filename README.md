# Fast Photo Mosaic
In this work, I implemented a photo mosaic algorithm based on feature matching.
I designed a feature descriptor based on the mean histogram of the LAB color space, applied the K-D tree to match the color blocks of the target image sub-region, and used the pre-computed feature pool to optimize the synthesis speed, and realized the mosaic photo that has better performance than *Foto-Mosaik-Edda*.

## Quick Start
1. Check out this repository and download the source code

    `git clone git@github.com:silvery107/fast-photo-mosaic.git`

2. Install the required python modules

    `pip install -r requirements.txt`

3. Start photo mosaic by `import photo_mosaic` and calling `mosaic(tgt_img_pth, tiles, types)` 

| Parameter | Description |
|:---|:---|
| tgt_img_pth | Directory path of target image. |
| tiles | The resolution of mosaic elements, and each values should be an **integer multiply of 8**. |
| types | Currently it support **two types**, natural and manmade, you can add new image types under `data/<your_type>` with corresponding `<your_type>.txt` description file. |


* For more usage details, please check `quick_start.ipynb`

## Composited Image Gallery
<p float="left">
  <img src=./images/composite1.png width="300" />
  <img src=./images/target1.jpg width="300" />
</p>


<p float="left">
  <img src=./images/composite2.png width="300" />
  <img src=./images/target2.jpg width="300" />
</p>

<p float="left">
  <img src=./images/composite3.png width="300" />
  <img src=./images/target3.jpg width="300" />
</p>

## Call Graph

<img src=./images/pycallgraph_v1.png width=600>
