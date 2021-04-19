import os
import cv2
import numpy as np
import skimage
import tqdm
import photomosaic as pm
from scipy.spatial import cKDTree
from skimage import img_as_float
dir_data = "./data/"
dir_pool = "./pools/"
dir_resize = "./data/"
dir_com = "./composition/"

# GUI interface method
def main(target_img_dir,patch_pix,mode=2):
    name_target = target_img_dir.split(sep='/')[-1]
    image = cv2.imread(target_img_dir)
    preprocess_source()
    dir_pool = None
    pool = None
    if mode == 2: # all
        fname = './data/manmade_training'
        pool1, dir_pool1 = get_pool_and_dir(fname)
        
        fname = './data/natural_training'
        pool2, dir_pool2 = get_pool_and_dir(fname)

        pool = np.concatenate((pool1,pool2),axis=0)
        dir_pool = dir_pool1+dir_pool2

    elif mode == 1: # manmade
        fname = './data/manmade_training'
        pool, dir_pool = get_pool_and_dir(fname)

    elif mode == 0: # natural
        fname = './data/natural_training'
        pool, dir_pool = get_pool_and_dir(fname)

    img_composited = get_composite(image,dir_pool,patch_pix,pool)
    cv2.imwrite('./composition/mosaic_'+name_target,img_composited)

# preprocessing source images methods
def get_pool_and_dir(fname):
    pool = get_pool(fname)
    cut_name = fname.split(sep='/')[-1]
    with open('resize_'+cut_name+'.txt') as manmade_training:
        dir_pool = manmade_training.read().splitlines()
    return pool, dir_pool

def preprocess_source():
    resize_source('./data/manmade_training','./data/resize_manmade_training')
    resize_source('./data/natural_training','./data/resize_natural_training')

def resize_source(dir_path,dst_path,multicrop=False):
    '''
    Resize and crop images in 'dir_path' to 256*256,
    and create a txt file contained their pathes.
    One image will be cut into 4 sub images
    '''
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
        with open(dir_path+'.txt') as dir_txt:
            sources = dir_txt.read().splitlines()
        file_resize_image = open(dst_path+'.txt','w')
        tq_dir_path = tqdm.tqdm(sources,desc='resizing sources')
        for image in tq_dir_path:
            filename = image.split(sep='/')[1]
            img = cv2.imread(dir_path+'/'+filename)
            if not multicrop:
                img = cv2.resize(img,(256,256),interpolation=cv2.INTER_AREA)
                cv2.imwrite(dst_path+'/'+filename,img)
                file_resize_image.write(dst_path+'/'+filename+'\n')
            else:
                img = cv2.resize(img,(384,384),interpolation=cv2.INTER_AREA)
                img_sub1 = img[:256,:256]
                img_sub2 = img[:256,128:]
                img_sub3 = img[128:,:256]
                img_sub4 = img[128:,128:]
                img_set = [img_sub1,img_sub2,img_sub3,img_sub4]
                for i in range(4):
                    cv2.imwrite(dst_path+'/'+filename+'_sub'+str(i)+'.jpg',img_set[i])
                    file_resize_image.write(dst_path+'/'+filename+'\n')

        file_resize_image.close()

def get_pool(dir_pool):
    folder = dir_pool.split(sep='/')[-1]
    pool_file_name = folder+'_pool.npy'
    with open(dir_pool+'.txt') as sources:
        img_list = sources.read().splitlines()

    if not os.path.isfile(pool_file_name):
        pool = []
        length = len(img_list)
        assert length>0
        for i in tqdm.trange(length,desc='making pool'):
            filename = img_list[i].split(sep='/')[-1]
            img = cv2.cvtColor(cv2.imread('./data/resize_'+folder+'/'+filename),cv2.COLOR_BGR2RGB)
            img_formated = format_image(img)
            feature = get_feature(img_formated)
            pool.append(feature)
        np.save(pool_file_name,pool)

    else:
        pool = np.load(pool_file_name)

    return pool

# compositing method
def get_composite(image,dir_pool,tiles,pool_cache=None):
    '''
    Get composite image, use the pre-generated pool will speed up composition
    ---
    image   : (m,n,c) ndarray, BGR image
    dir_pool: (k,) list of str, directories
    tiles   : (a,b) tuple, size of mosaic patches
    pool_cache: (k,48) list, features if pool has already generated
    ---
    return: img_composite,pool
    '''
    tile_h,tile_w = tiles
    img_h,img_w,channels = image.shape
    img_size = np.array(image.shape[:2])
    # adaptive resize
    while True:
        if max(img_size)>1500:
            img_size = np.multiply(img_size,0.9).astype('int32')
        elif max(img_size)<1000:
            img_size = np.multiply(img_size,1.1).astype('int32')
        else:
            break

    img_h,img_w = img_size
    img_resized = cv2.resize(image,(img_w,img_h))
    crop_size = (img_h-img_h%tile_h,img_w-img_w%tile_w)
    img_cropped = crop_image(img_resized,crop_size)
    img_rgb = cv2.cvtColor(img_cropped,cv2.COLOR_BGR2RGB) # read RGB image
    img_formated = format_image(img_rgb) # convert color space
    img_filted = cv2.GaussianBlur(img_formated,(9,9),2,borderType=cv2.BORDER_REPLICATE)
    # setup parameters
    img_h,img_w,channels = img_filted.shape
    # make pool
    if pool_cache is not None:
        pool = pool_cache
    else:
        pool = get_pool(dir_pool)
    # create hybrid image
    img_hybrid = np.zeros_like(img_filted)
    # match tiles
    kd_match = kd_matcher(pool)
    for x in tqdm.trange(0,img_h,tile_h,desc='matching tiles'):
        for y in range(0,img_w,tile_w):
            temp = img_filted[x:x+tile_h,y:y+tile_w]
            feature = get_feature(temp)
            k_index = kd_match(feature)
            sample = cv2.imread(dir_pool[k_index])
            sample = cv2.resize(sample,(tile_w,tile_h),interpolation=cv2.INTER_AREA)
            img_hybrid[x:x+tile_h,y:y+tile_w] =sample.astype('float32')

    img_blend = blend_grid(img_hybrid.astype('uint8'),tiles,d=round(min(tiles)*0.2))
    img_feather = feather_image(img_cropped,img_blend,alpha=0.3,mode='global')

    return img_feather.astype('uint8')

def get_feature(image):
    '''
    Get a 64*64 dimensions feature, suppose the size of 'image' are integer multiples of 8
    '''
    h,w = image.shape[:2]
    assert h%8==0 and w%8==0
    features = []
    # 16*16 cell
    step_h = int(h/8)
    step_w = int(w/8)
    for x in range(0,h,step_h):
        for y in range(0,w,step_w):
            features.append(np.mean(image[x:x+step_h,y:y+step_w].reshape(-1,3),0))

    feature = np.array(features)

    return np.reshape(feature,(-1))

# enhancing resute methods
def blend_grid(image,tiles,d=5):
    '''
    Blend each edges of the mosaic grid by 'd' pixels
        --|--|--
        --|--|-- 2*2 grid
    '''
    img_h,img_w = image.shape[:2]
    step_h,step_w = tiles
    img_blend = np.zeros_like(image)
    for x in range(step_h,img_h,step_h):
        for i in range(d):
            alpha = (d-i)/d
            img_blend[x-d+i,:] = image[x-d+i,:]*alpha+image[x+d-i,:]*(1-alpha)

    for y in range(step_w,img_w,step_w):
        for i in range(d):
            alpha = (d-i)/d
            img_blend[:,y-d+i] = image[:,y-d+i]*alpha+image[:,y+d-i]*(1-alpha)

    img_blend_gray = cv2.cvtColor(img_blend,cv2.COLOR_BGR2GRAY)
    thr,mask = cv2.threshold(img_blend_gray,1,255,cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    img_bg = cv2.bitwise_and(image,image,mask=mask_inv)

    return img_bg + img_blend

def feather_image(image1,image2,alpha=0.3,mode='global'):
    '''
    'global': add 'image1' to 'image2' by proportion of 'alpha'
    'corner': add Harris corners by proportion
    'edge'  : add Canny edges by proportion
    '''
    if image1.shape != image2.shape:
        try:
            image2 = crop_image(image2,image1.shape[:2])
        except AssertionError:
            image1 = crop_image(image1,image2.shape[:2])

    if mode == 'global':
        return (image1*alpha+image2*(1-alpha)).astype('uint8')

    if mode == 'corner':
        temp = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
        img1_harris = cv2.cornerHarris(temp,32,5,0.05)
        thr,mask = cv2.threshold(img1_harris,0,255,cv2.THRESH_BINARY)
    elif mode == 'edge':
        mask = cv2.Canny(image1,100,200)
    
    mask = mask.astype('uint8')
    mask_inv = cv2.bitwise_not(mask)
    image1_ = cv2.bitwise_and(image1,image1,mask=mask)
    image2_mask = cv2.bitwise_and(image2,image2,mask=mask)
    image2_ = cv2.bitwise_and(image2,image2,mask=mask_inv)
    return image2_mask*(1-alpha)+image2_

# standardization methods
def crop_image(image,crop_size):
    img_h,img_w = image.shape[:2]
    c_h,c_w = crop_size
    assert img_h>=c_h and img_w>=c_w
    temp1 = img_h-c_h
    temp2 = img_w-c_w
    c_h = int((temp1+1)//2)
    c_w = int((temp2+1)//2)

    return image[c_h:img_h-temp1+c_h,c_w:img_w-temp2+c_w]

def format_image(image):
    '''
    Convert 'image' to LAB color space
    '''
    # image -> [0,1] -> perceptually-uniform color space
    img_float = skimage.img_as_float32(image)
    img_perceptual = cv2.cvtColor(img_float,cv2.COLOR_RGB2Lab)

    return img_perceptual.astype('float32')

# matching method
def kd_matcher(data):
    tree = cKDTree(data)
    def match(vector):
        _,index = tree.query(vector,k=1)
        return index

    return match

def pm_test(img_target,dir_pool,grid_dims,depth=1):
    image = pm.imread(img_target)
    pool = pm.make_pool(dir_pool)
    image = img_as_float(image)
    converted_img = pm.perceptual(image) # change color space
    scaled_img = pm.rescale_commensurate(converted_img, grid_dims=grid_dims, depth=depth)
    tiles = pm.partition(scaled_img, grid_dims=grid_dims, depth=depth)
    tile_colors = [np.mean(scaled_img[tile].reshape(-1,image.shape[-1]), 0) for tile in tiles]
    match = pm.simple_matcher(pool) # kd tree
    matches = [match(tc) for tc in tile_colors]
    canvas = np.ones_like(scaled_img)  # white canvas
    mos = pm.draw_mosaic(canvas, tiles, matches)

    return mos

if __name__ == '__main__':
    # from pycallgraph import PyCallGraph
    # from pycallgraph.output import GraphvizOutput
    # with PyCallGraph(output=GraphvizOutput()):
        target_img_dir = 'mosaic_target1.jpg'
        patch_pix = (32,32) # note that each values should be integer multiply of 8
        mode = 2
        main(target_img_dir,patch_pix,mode)
