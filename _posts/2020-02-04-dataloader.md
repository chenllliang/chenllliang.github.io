---
layout:     post
title:      Pytorch之Dataset与DataLoader
subtitle:   打造你自己的数据集,源码阅读
date:       2020-02-04
author:     CL
header-img: img/tag-bg-o.jpg
catalog: true
tags:
    - PyTorch
---

深度时代，数据为王。

PyTorch为我们提供的两个Dataset和DataLoader类分别负责可被Pytorhc使用的数据集的创建以及向训练传递数据的任务。如果想个性化自己的数据集或者数据传递方式，也可以自己重写子类。

Dataset是DataLoader实例化的一个参数，所以这篇文章会先从Dataset的源代码讲起，然后讲到DataLoader，关注主要函数，少细枝末节，目的是使大家学会自定义自己的数据集。

# Dataset

## 什么时候使用Dataset
CIFAR10是CV训练中经常使用到的一个数据集，在PyTorch中CIFAR10是一个写好的Dataset，我们使用时只需以下代码：
```python
data = datasets.CIFAR10("./data/", transform=transform, train=True, download=True)。
```
datasets.CIFAR10就是一个Datasets子类，data是这个类的一个实例。

我们有的时候需要用自己在一个文件夹中的数据作为数据集，这个时候，我们可以使用ImageFolder这个方便的API。
```python
FaceDataset = datasets.ImageFolder('./data', transform=img_transform)
```

## 如何自定义一个数据集
**torch.utils.data. ataset** 是一个表示数据集的抽象类。任何自定义的数据集都需要继承这个类并覆写相关方法。

所谓数据集，其实就是一个负责处理索引(index)到样本(sample)映射的一个类(class)。

Pytorch提供两种数据集：
* Map式数据集
* Iterable式数据集


### Map式数据集

一个Map式的数据集必须要重写__getitem__(self, index),__len__(self) 两个内建方法，用来表示从索引到样本的映射（Map）.

这样一个数据集dataset，举个例子，当使用dataset[idx]命令时，可以在你的硬盘中读取你的数据集中第idx张图片以及其标签（如果有的话）;len(dataset)则会返回这个数据集的容量。

例子-1：
```python
class CustomDataset(data.Dataset):#需要继承data.Dataset
    def __init__(self):
        # TODO
        # 1. Initialize file path or list of file names.
    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        #这里需要注意的是，第一步：read one data，是一个data
        pass
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return 0

```
自己实验中写的一个例子：这里我们的图片文件储存在“./data/faces/”文件夹下，图片的名字并不是从1开始，而是从final_train_tag_dict.txt这个文件保存的字典中读取，label信息也是用这个文件中读取。大家可以照着上面的注释阅读这段代码。
```python
from torch.utils import data
import numpy as np
from PIL import Image


class face_dataset(data.Dataset):
	def __init__(self):
		self.file_path = './data/faces/'
		f=open("final_train_tag_dict.txt","r")
		self.label_dict=eval(f.read())
		f.close()

	def __getitem__(self,index):
		label = list(self.label_dict.values())[index-1]
		img_id = list(self.label_dict.keys())[index-1]
		img_path = self.file_path+str(img_id)+".jpg"
		img = np.array(Image.open(img_path))
		return img,label

	def __len__(self):
		return len(self.label_dict)

```

下面我们看一下官方MNIST数据集的例子

```python
class MNIST(data.Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']
    class_to_idx = {_class: i for i, _class in enumerate(classes)}

    @property
    def targets(self):
        if self.train:
            return self.train_labels
        else:
            return self.test_labels

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            self.train_data, self.train_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file))
        else:
            self.test_data, self.test_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import gzip

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print('Processing...')

        training_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

```

### Iterable式数据集
一个Iterable（迭代）式数据集是抽象类**data.IterableDataset**的子类，并且覆写了__iter__方法成为一个迭代器。这种数据集主要用于数据大小未知，或者以流的形式的输入，本地文件不固定的情况，需要以迭代的方式来获取样本索引。

关于迭代器与生成器的知识可以参见博主的另一篇文章[Python迭代器与生成器介绍及在Pytorch源码中应用](https://chenllliang.github.io/2020/02/06/PyIter/)。

这一块先mark着，因为还没有找到应用场景。



# DataLoader
