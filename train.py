from model import *
from dataset import *

import torch
import torch.nn as nn

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from statistics import mean


class Train:
    def __init__(self, args):
        self.mode = args.mode
        self.train_continue = args.train_continue

        self.scope = args.scope
        self.norm = args.norm

        self.dir_checkpoint = args.dir_checkpoint
        self.dir_log = args.dir_log

        self.name_data = args.name_data
        self.dir_data = args.dir_data
        self.dir_result = args.dir_result

        self.num_epoch = args.num_epoch
        self.batch_size = args.batch_size

        self.lr_G = args.lr_G

        self.optim = args.optim
        self.beta1 = args.beta1

        self.ny_in = args.ny_in
        self.nx_in = args.nx_in
        self.nch_in = args.nch_in

        self.ny_load = args.ny_load
        self.nx_load = args.nx_load
        self.nch_load = args.nch_load

        self.ny_out = args.ny_out
        self.nx_out = args.nx_out
        self.nch_out = args.nch_out

        self.nch_ker = args.nch_ker

        self.data_type = args.data_type

        self.num_freq_disp = args.num_freq_disp
        self.num_freq_save = args.num_freq_save

        self.gpu_ids = args.gpu_ids

        if self.gpu_ids and torch.cuda.is_available():
            self.device = torch.device("cuda:%d" % self.gpu_ids[0])
            torch.cuda.set_device(self.gpu_ids[0])
        else:
            self.device = torch.device("cpu")

    def save(self, dir_chck, netG, optimG, epoch):
        if not os.path.exists(dir_chck):
            os.makedirs(dir_chck)

        torch.save({'netG': netG.state_dict(),
                    'optimG': optimG.state_dict()},
                   '%s/model_epoch%04d.pth' % (dir_chck, epoch))

    def load(self, dir_chck, netG, optimG=[], epoch=[], mode='train'):
        if not epoch:
            ckpt = os.listdir(dir_chck)
            ckpt.sort()
            epoch = int(ckpt[-1].split('epoch')[1].split('.pth')[0])

        dict_net = torch.load('%s/model_epoch%04d.pth' % (dir_chck, epoch))

        print('Loaded %dth network' % epoch)

        if mode == 'train':
            netG.load_state_dict(dict_net['netG'])
            optimG.load_state_dict(dict_net['optimG'])

            return netG, optimG, epoch

        elif mode == 'test':
            netG.load_state_dict(dict_net['netG'])

            return netG, epoch

    def preprocess(self, data):
        normalize = Normalize()
        randflip = RandomFlip()
        rescale = Rescale((self.ny_load, self.nx_load))
        randomcrop = RandomCrop((self.ny_out, self.nx_out))
        totensor = ToTensor()
        return totensor(randomcrop(rescale(randflip(normalize(data)))))

    def deprocess(self, data):
        tonumpy = ToNumpy()
        denomalize = Denomalize()
        return denomalize(tonumpy(data))

    def train(self):
        mode = self.mode

        train_continue = self.train_continue
        num_epoch = self.num_epoch

        lr_G = self.lr_G

        batch_size = self.batch_size
        device = self.device

        gpu_ids = self.gpu_ids

        nch_in = self.nch_in
        nch_out = self.nch_out
        nch_ker = self.nch_ker

        norm = self.norm
        name_data = self.name_data

        num_freq_disp = self.num_freq_disp
        num_freq_save = self.num_freq_save

        ## setup dataset
        dir_chck = os.path.join(self.dir_checkpoint, self.scope, name_data)

        dir_data_train = os.path.join(self.dir_data, name_data, 'train')
        dir_data_val = os.path.join(self.dir_data, name_data, 'val')

        dir_log_train = os.path.join(self.dir_log, self.scope, name_data, 'train')
        dir_log_val = os.path.join(self.dir_log, self.scope, name_data, 'val')

        transform_train = transforms.Compose([Normalize(), RandomFlip(), RandomCrop((self.ny_in, self.nx_in)), ToTensor()])
        transform_val = transforms.Compose([Normalize(), RandomFlip(), RandomCrop((self.ny_in, self.nx_in)), ToTensor()])
        # transform_val = transforms.Compose([Normalize(), ToTensor()])
        transform_inv = transforms.Compose([ToNumpy(), Denomalize()])
        transform_ts2np = ToNumpy()

        dataset_train = Dataset(dir_data_train, data_type=self.data_type, nch=self.nch_in, transform=transform_train)
        dataset_val = Dataset(dir_data_val, data_type=self.data_type, nch=self.nch_in, transform=transform_val)

        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)
        loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=8)

        num_train = len(dataset_train)
        num_val = len(dataset_val)

        num_batch_train = int((num_train / batch_size) + ((num_train % batch_size) != 0))
        num_batch_val = int((num_val / batch_size) + ((num_val % batch_size) != 0))

        ## setup network
        netG = UNet(nch_in, nch_out, nch_ker, norm)

        init_net(netG, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)

        ## setup loss & optimization
        # fn_L1 = nn.L1Loss().to(device) # L1
        # fn_CLS = nn.BCEWithLogitsLoss().to(device)
        fn_CLS = nn.BCELoss().to(device)

        paramsG = netG.parameters()

        optimG = torch.optim.Adam(paramsG, lr=lr_G, betas=(self.beta1, 0.999))

        ## load from checkpoints
        st_epoch = 0

        if train_continue == 'on':
            netG, optimG, st_epoch = self.load(dir_chck, netG, optimG, mode=mode)

        ## setup tensorboard
        writer_train = SummaryWriter(log_dir=dir_log_train)
        writer_val = SummaryWriter(log_dir=dir_log_val)

        for epoch in range(st_epoch + 1, num_epoch + 1):
            ## training phase
            netG.train()

            loss_G_cls_train = []

            for i, data in enumerate(loader_train, 1):
                def should(freq):
                    return freq > 0 and (i % freq == 0 or i == num_batch_train)

                input = data['input'].to(device)
                label = data['label'].to(device)

                # forward netG
                output = netG(input)

                # backward netG
                optimG.zero_grad()

                loss_G_cls = fn_CLS(output, label)

                loss_G_cls.backward()
                optimG.step()

                # get losses
                loss_G_cls_train += [loss_G_cls.item()]

                print('TRAIN: EPOCH %d: BATCH %04d/%04d: CLS: %.4f'
                      % (epoch, i, num_batch_train, mean(loss_G_cls_train)))

                if should(num_freq_disp):
                    ## show output
                    input = transform_inv(input)
                    label = transform_ts2np(label)

                    # output = transform_inv(output)
                    output = transform_ts2np(output)
                    output = 1.0 * (output > 0.5)

                    writer_train.add_images('input', input, num_batch_train * (epoch - 1) + i, dataformats='NHWC')
                    writer_train.add_images('output', output, num_batch_train * (epoch - 1) + i, dataformats='NHWC')
                    writer_train.add_images('label', label, num_batch_train * (epoch - 1) + i, dataformats='NHWC')

            writer_train.add_scalar('loss_G_cls', mean(loss_G_cls_train), epoch)

            ## validation phase
            with torch.no_grad():
                netG.eval()

                loss_G_cls_val = []

                for i, data in enumerate(loader_val, 1):
                    def should(freq):
                        return freq > 0 and (i % freq == 0 or i == num_batch_val)

                    input = data['input'].to(device)
                    label = data['label'].to(device)

                    # forward netG
                    output = netG(input)

                    loss_G_cls = fn_CLS(output, label)

                    loss_G_cls_val += [loss_G_cls.item()]

                    print('VALID: EPOCH %d: BATCH %04d/%04d: CLS: %.4f'
                          % (epoch, i, num_batch_val, mean(loss_G_cls_val)))

                    if should(num_freq_disp):
                        ## show output
                        input = transform_inv(input)
                        label = transform_ts2np(label)

                        # output = transform_inv(output)
                        output = transform_ts2np(output)
                        output = 1.0 * (output > 0.5)

                        writer_val.add_images('input', input, num_batch_val * (epoch - 1) + i, dataformats='NHWC')
                        writer_val.add_images('output', output, num_batch_val * (epoch - 1) + i, dataformats='NHWC')
                        writer_val.add_images('label', label, num_batch_val * (epoch - 1) + i, dataformats='NHWC')

                writer_val.add_scalar('loss_G_cls', mean(loss_G_cls_val), epoch)

            # update schduler
            # schedG.step()
            # schedD.step()

            ## save
            if (epoch % num_freq_save) == 0:
                self.save(dir_chck, netG, optimG, epoch)

        writer_train.close()
        writer_val.close()

    def test(self):
        mode = self.mode

        batch_size = self.batch_size
        device = self.device
        gpu_ids = self.gpu_ids

        nch_in = self.nch_in
        nch_out = self.nch_out
        nch_ker = self.nch_ker

        norm = self.norm

        name_data = self.name_data

        if nch_out == 1:
            cmap = 'gray'
        else:
            cmap = None

        ## setup dataset
        dir_chck = os.path.join(self.dir_checkpoint, self.scope, name_data)

        dir_result = os.path.join(self.dir_result, self.scope, name_data)
        dir_result_save = os.path.join(dir_result, 'images')
        if not os.path.exists(dir_result_save):
            os.makedirs(dir_result_save)

        dir_data_test = os.path.join(self.dir_data, name_data, 'test')

        transform_test = transforms.Compose([Normalize(), ToTensor()])
        transform_inv = transforms.Compose([ToNumpy(), Denomalize()])
        transform_ts2np = ToNumpy()

        dataset_test = Dataset(dir_data_test, data_type=self.data_type, nch=self.nch_in, transform=transform_test)

        loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8)

        num_test = len(dataset_test)

        num_batch_test = int((num_test / batch_size) + ((num_test % batch_size) != 0))

        ## setup network
        netG = UNet(nch_in, nch_out, nch_ker, norm)
        init_net(netG, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)

        ## setup loss & optimization
        # fn_L1 = nn.L1Loss().to(device)  # L1
        # fn_CLS = nn.BCEWithLogitsLoss().to(device)
        fn_CLS = nn.BCELoss().to(device)

        ## load from checkpoints
        st_epoch = 0

        netG, st_epoch = self.load(dir_chck, netG, mode=mode)

        ## test phase
        with torch.no_grad():
            netG.eval()
            # netG.train()

            loss_G_cls_test = []

            for i, data in enumerate(loader_test, 1):
                input = data['input'].to(device)
                label = data['label'].to(device)

                output = netG(input)

                loss_G_cls = fn_CLS(output, label)

                loss_G_cls_test += [loss_G_cls.item()]

                input = transform_inv(input)
                label = transform_ts2np(label)

                # output = transform_inv(output)
                output = transform_ts2np(output)
                output = 1.0 * (output > 0.5)

                for j in range(label.shape[0]):
                    name = batch_size * (i - 1) + j
                    fileset = {'name': name,
                               'input': "%04d-input.png" % name,
                               'output': "%04d-output.png" % name,
                               'label': "%04d-label.png" % name}

                    plt.imsave(os.path.join(dir_result_save, fileset['input']), input[j, :, :, :].squeeze(), cmap=cmap)
                    plt.imsave(os.path.join(dir_result_save, fileset['output']), output[j, :, :, :].squeeze(), cmap=cmap)
                    plt.imsave(os.path.join(dir_result_save, fileset['label']), label[j, :, :, :].squeeze(), cmap=cmap)

                    append_index(dir_result, fileset)

                print('TEST: %d/%d: LOSS: %.6f' % (i, num_batch_test, loss_G_cls.item()))
            print('TEST: AVERAGE LOSS: %.6f' % (mean(loss_G_cls_test)))


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def append_index(dir_result, fileset, step=False):
    index_path = os.path.join(dir_result, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        for key, value in fileset.items():
            index.write("<th>%s</th>" % key)
        index.write('</tr>')

    # for fileset in filesets:
    index.write("<tr>")

    if step:
        index.write("<td>%d</td>" % fileset["step"])
    index.write("<td>%s</td>" % fileset["name"])

    del fileset['name']

    for key, value in fileset.items():
        index.write("<td><img src='images/%s'></td>" % value)

    index.write("</tr>")
    return index_path


def add_plot(output, label, writer, epoch=[], ylabel='Density', xlabel='Radius', namescope=[]):
    fig, ax = plt.subplots()

    ax.plot(output.transpose(1, 0).detach().numpy(), '-')
    ax.plot(label.transpose(1, 0).detach().numpy(), '--')

    ax.set_xlim(0, 400)

    ax.grid(True)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    writer.add_figure(namescope, fig, epoch)
