import numpy as np
import torch
import skimage
from skimage import transform
import matplotlib.pyplot as plt
import os


class Dataset(torch.utils.data.Dataset):
    """
    dataset of image files of the form 
       stuff<number>_trans.pt
       stuff<number>_density.pt
    """

    def __init__(self, data_dir, data_type='float32', nch=3, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data_type = data_type
        self.nch = nch

        lst_data = os.listdir(data_dir)

        lst_input = [f for f in lst_data if f.startswith('input')]
        lst_label = [f for f in lst_data if f.startswith('label')]

        lst_input.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        lst_label.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        self.lst_input = lst_input
        self.lst_label = lst_label

    def __getitem__(self, index):
        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        input = np.load(os.path.join(self.data_dir, self.lst_input[index]))

        if label.dtype == np.uint8:
            label = label / 255.0

        if input.dtype == np.uint8:
            input = input / 255.0

        if label.ndim == 2:
            label = np.expand_dims(label, axis=2)

        if input.ndim == 2:
            input = np.expand_dims(input, axis=2)

        data = {'input': input, 'label': label}

        if self.transform:
            data = self.transform(data)

        return data

    def __len__(self):
        return len(self.lst_label)


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):
        # Swap color axis because numpy image: H x W x C
        #                         torch image: C x H x W

        # for key, value in data:
        #     data[key] = torch.from_numpy(value.transpose((2, 0, 1)))
        #
        # return data

        input, label = data['input'], data['label']

        input = input.transpose((2, 0, 1)).astype(np.float32)
        label = label.transpose((2, 0, 1)).astype(np.float32)
        return {'input': torch.from_numpy(input), 'label': torch.from_numpy(label)}


class Normalize(object):
    def __call__(self, data):
        # Nomalize [0, 1] => [-1, 1]

        # for key, value in data:
        #     data[key] = 2 * (value / 255) - 1
        #
        # return data

        input, label = data['input'], data['label']
        input = 2 * input - 1
        # dataB = 2 * dataB - 1
        return {'input': input, 'label': label}


class RandomFlip(object):
    def __call__(self, data):
        # Random Left or Right Flip

        # for key, value in data:
        #     data[key] = 2 * (value / 255) - 1
        #
        # return data
        input, label = data['input'], data['label']

        if np.random.rand() > 0.5:
            input = np.fliplr(input)
            label = np.fliplr(label)

        if np.random.rand() > 0.5:
            input = np.flipud(input)
            label = np.flipud(label)

        return {'input': input, 'label': label}


class Rescale(object):
  """Rescale the image in a sample to a given size

  Args:
    output_size (tuple or int): Desired output size.
                                If tuple, output is matched to output_size.
                                If int, smaller of image edges is matched
                                to output_size keeping aspect ratio the same.
  """

  def __init__(self, output_size):
    assert isinstance(output_size, (int, tuple))
    self.output_size = output_size

  def __call__(self, data):
    input, label = data['input'], data['label']

    h, w = input.shape[:2]

    if isinstance(self.output_size, int):
      if h > w:
        new_h, new_w = self.output_size * h / w, self.output_size
      else:
        new_h, new_w = self.output_size, self.output_size * w / h
    else:
      new_h, new_w = self.output_size

    new_h, new_w = int(new_h), int(new_w)

    input = transform.resize(input, (new_h, new_w))
    label = transform.resize(label, (new_h, new_w))

    return {'input': input, 'label': label}


class RandomCrop(object):
  """Crop randomly the image in a sample

  Args:
    output_size (tuple or int): Desired output size.
                                If int, square crop is made.
  """

  def __init__(self, output_size):
    assert isinstance(output_size, (int, tuple))
    if isinstance(output_size, int):
      self.output_size = (output_size, output_size)
    else:
      assert len(output_size) == 2
      self.output_size = output_size

  def __call__(self, data):
    input, label = data['input'], data['label']

    h, w = input.shape[:2]
    new_h, new_w = self.output_size

    top = np.random.randint(0, h - new_h)
    left = np.random.randint(0, w - new_w)

    input = input[top: top + new_h, left: left + new_w]
    label = label[top: top + new_h, left: left + new_w]

    return {'input': input, 'label': label}


class ToNumpy(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):
        # Swap color axis because numpy image: H x W x C
        #                         torch image: C x H x W

        # for key, value in data:
        #     data[key] = value.transpose((2, 0, 1)).numpy()
        #
        # return data

        return data.to('cpu').detach().numpy().transpose(0, 2, 3, 1)

        # input, label = data['input'], data['label']
        # input = input.transpose((2, 0, 1))
        # label = label.transpose((2, 0, 1))
        # return {'input': input.detach().numpy(), 'label': label.detach().numpy()}


class Denomalize(object):
    def __call__(self, data):
        # Denomalize [-1, 1] => [0, 1]

        # for key, value in data:
        #     data[key] = (value + 1) / 2 * 255
        #
        # return data

        return (data + 1) / 2

        # input, label = data['input'], data['label']
        # input = (input + 1) / 2 * 255
        # label = (label + 1) / 2 * 255
        # return {'input': input, 'label': label}
