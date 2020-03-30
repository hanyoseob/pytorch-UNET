from model import *
from dataset import *

import torch
import torch.nn as nn

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian

##
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

        if self.gpu_ids:
            torch.save({'netG': netG.module.state_dict(),
                        'optimG': optimG.state_dict()},
                        '%s/model_epoch%04d.pth' % (dir_chck, epoch))
        else:
            torch.save({'netG': netG.state_dict(),
                        'optimG': optimG.state_dict()},
                        '%s/model_epoch%04d.pth' % (dir_chck, epoch))


    def load(self, dir_chck, netG, optimG=None, epoch=None):
        if not os.path.exists(dir_chck):
            epoch = 0
            if optimG is None:
                return netG, epoch
            else:
                return netG, optimG, epoch

        if not epoch:
            ckpt = os.listdir(dir_chck)
            ckpt = [f for f in ckpt if f.startswith('model')]
            ckpt.sort()

            epoch = int(ckpt[-1].split('epoch')[1].split('.pth')[0])

        dict_net = torch.load('%s/model_epoch%04d.pth' % (dir_chck, epoch), map_location=self.device)

        print('Loaded %dth network' % epoch)

        if optimG is None:
            netG.load_state_dict(dict_net['netG'])
            return netG, epoch
        else:
            netG.load_state_dict(dict_net['netG'])
            optimG.load_state_dict(dict_net['optimG'])
            return netG, optimG, epoch


    def train(self):
        mode = self.mode

        train_continue = self.train_continue
        num_epoch = self.num_epoch

        lr_G = self.lr_G
        beta1 = self.beta1

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

        cmap = 'gray' if nch_out == 1 else None

        ## setup dataset
        dir_chck = os.path.join(self.dir_checkpoint, self.scope, name_data)

        dir_result_train = os.path.join(self.dir_result, self.scope, name_data, 'train')
        if not os.path.exists(os.path.join(dir_result_train, 'images')):
            os.makedirs(os.path.join(dir_result_train, 'images'))

        dir_result_val = os.path.join(self.dir_result, self.scope, name_data, 'val')
        if not os.path.exists(os.path.join(dir_result_val, 'images')):
            os.makedirs(os.path.join(dir_result_val, 'images'))

        dir_data_train = os.path.join(self.dir_data, name_data, 'train')
        dir_data_val = os.path.join(self.dir_data, name_data, 'val')

        dir_log_train = os.path.join(self.dir_log, self.scope, name_data, 'train')
        dir_log_val = os.path.join(self.dir_log, self.scope, name_data, 'val')

        transform_train = transforms.Compose([RandomCrop((self.ny_load, self.nx_load)), RandomFlip(), Normalize(), ToTensor()])
        transform_val = transforms.Compose([RandomFlip(), Normalize(), ToTensor()])

        transform_inv = transforms.Compose([ToNumpy(), Denormalize()])
        transform_ts2np = ToNumpy()

        dataset_train = Dataset(dir_data_train, data_type=self.data_type, transform=transform_train)
        dataset_val = Dataset(dir_data_val, data_type=self.data_type, transform=transform_val)

        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)
        loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=8)

        num_train = len(dataset_train)
        num_val = len(dataset_val)

        num_batch_train = int((num_train / batch_size) + ((num_train % batch_size) != 0))
        num_batch_val = int((num_val / batch_size) + ((num_val % batch_size) != 0))

        ## setup network
        netG = UNet(nch_in, nch_out, nch_ker, norm)
        # netG = CNP(nch_in, nch_out, nch_ker, norm)

        init_weights(netG, init_type='normal', init_gain=0.02)
        netG.to(device)

        paramsG = netG.parameters()
        optimG = torch.optim.Adam(paramsG, lr=lr_G, betas=(beta1, 0.999))

        ## load from checkpoints
        st_epoch = 0

        if train_continue == 'on':
            netG, optimG, st_epoch = self.load(dir_chck, netG, optimG)

        if gpu_ids:
            netG = torch.nn.DataParallel(netG, gpu_ids)  # multi-GPUs
            # for state in optimG.state.values():
            #     for k, v in state.items():
            #         if isinstance(v, torch.Tensor):
            #             state[k] = v.cuda()

        ## setup loss & optimization
        # fn_L1 = nn.L1Loss().to(device)      # Regression loss: L1
        # fn_L2 = nn.MSELoss().to(device)     # Regression loss: L2

        # fn_CLS = nn.BCELoss().to(device)
        # fn_CLS = nn.NLLLoss().to(device)

        fn_CLS = nn.BCEWithLogitsLoss().to(device)    # Binary-class: This loss combines a `Sigmoid` layer and the `BCELoss` in one single class.
        # fn_CLS = nn.CrossEntropyLoss().to(device)     # Multi-class: This criterion combines :func:`nn.LogSoftmax` and :func:`nn.NLLLoss` in one single class.

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
                      % (epoch, i, num_batch_train, np.mean(loss_G_cls_train)))

                if should(num_freq_disp):
                    ## show output
                    input = transform_inv(input)
                    label = transform_ts2np(label)
                    output = transform_ts2np(torch.sigmoid(output))
                    output = 1.0 * (output > 0.5)

                    # writer_train.add_images('input', input, num_batch_train * (epoch - 1) + i, dataformats='NHWC')
                    # writer_train.add_images('output', output, num_batch_train * (epoch - 1) + i, dataformats='NHWC')
                    # writer_train.add_images('label', label, num_batch_train * (epoch - 1) + i, dataformats='NHWC')

                    name = num_train * (epoch - 1) + num_batch_train * (i - 1)

                    fileset = {'name': name,
                               'input': "%06d-input.png" % name,
                               'label': "%06d-label.png" % name,
                               'output': "%06d-output.png" % name}

                    plt.imsave(os.path.join(dir_result_train, 'images', fileset['input']), input[0].squeeze(), cmap=cmap)
                    plt.imsave(os.path.join(dir_result_train, 'images', fileset['label']), label[0].squeeze(), cmap=cmap)
                    plt.imsave(os.path.join(dir_result_train, 'images', fileset['output']), output[0].squeeze(), cmap=cmap)

            writer_train.add_scalar('loss_G_cls', np.mean(loss_G_cls_train), epoch)

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
                          % (epoch, i, num_batch_val, np.mean(loss_G_cls_val)))

                    if should(num_freq_disp):
                        ## show output
                        input = transform_inv(input)
                        label = transform_ts2np(label)
                        output = transform_ts2np(torch.sigmoid(output))
                        output = 1.0 * (output > 0.5)

                        # writer_val.add_images('input', input, num_batch_val * (epoch - 1) + i, dataformats='NHWC')
                        # writer_val.add_images('output', output, num_batch_val * (epoch - 1) + i, dataformats='NHWC')
                        # writer_val.add_images('label', label, num_batch_val * (epoch - 1) + i, dataformats='NHWC')

                        name = num_val * (epoch - 1) + num_batch_val * (i - 1)

                        fileset = {'name': name,
                                   'input': "%06d-input.png" % name,
                                   'label': "%06d-label.png" % name,
                                   'output': "%06d-output.png" % name}

                        plt.imsave(os.path.join(dir_result_val, 'images', fileset['input']), input[0].squeeze(), cmap=cmap)
                        plt.imsave(os.path.join(dir_result_val, 'images', fileset['label']), label[0].squeeze(), cmap=cmap)
                        plt.imsave(os.path.join(dir_result_val, 'images', fileset['output']), output[0].squeeze(), cmap=cmap)


                writer_val.add_scalar('loss_G_cls', np.mean(loss_G_cls_val), epoch)

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

        cmap = 'gray' if nch_out == 1 else None

        ## setup dataset
        dir_chck = os.path.join(self.dir_checkpoint, self.scope, name_data)

        dir_result_test = os.path.join(self.dir_result, self.scope, name_data, 'test')
        dir_result_test_save = os.path.join(dir_result_test, 'images')
        if not os.path.exists(dir_result_test_save):
            os.makedirs(dir_result_test_save)

        dir_data_test = os.path.join(self.dir_data, name_data, 'test')

        transform_test = transforms.Compose([Normalize(), ToTensor()])
        transform_inv = transforms.Compose([ToNumpy(), Denormalize()])
        transform_ts2np = ToNumpy()

        dataset_test = Dataset(dir_data_test, data_type=self.data_type, transform=transform_test)

        loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8)

        num_test = len(dataset_test)

        num_batch_test = int((num_test / batch_size) + ((num_test % batch_size) != 0))

        ## setup network
        netG = UNet(nch_in, nch_out, nch_ker, norm)
        # netG = CNP(nch_in, nch_out, nch_ker, norm)

        init_weights(netG, init_type='normal', init_gain=0.02)
        netG.to(device)

        st_epoch = 0
        netG, st_epoch = self.load(dir_chck, netG)

        if gpu_ids:
            netG = torch.nn.DataParallel(netG, gpu_ids)  # multi-GPUs

        ## setup loss & optimization
        # fn_L1 = nn.L1Loss().to(device)  # L1
        # fn_CLS = nn.BCELoss().to(device)
        fn_CLS = nn.BCEWithLogitsLoss().to(device)

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
                output = transform_ts2np(torch.sigmoid(output))


                for j in range(label.shape[0]):
                    name = batch_size * (i - 1) + j

                    fileset = {'name': name,
                               'input': "%04d-input.png" % name,
                               'output_th': "%04d-output_th.png" % name,
                               'output_crf': "%04d-output_crf.png" % name,
                               'label': "%04d-label.png" % name}

                    input_ = input[j]
                    output_ = output[j]
                    label_ = label[j]

                    output_th = 1.0 * (output_ > 0.5)

                    ##
                    sdims = 1
                    schan = 1
                    compat = 10
                    iter = 10

                    output_crf = output_.transpose((2, 0, 1))
                    output_crf = np.concatenate((1 - output_crf, output_crf), axis=0)

                    U = unary_from_softmax(output_crf)

                    d = dcrf.DenseCRF2D(output_crf.shape[1], output_crf.shape[2], output_crf.shape[0])
                    d.setUnaryEnergy(U)


                    pairwise_energy = create_pairwise_bilateral(sdims=(sdims, sdims), schan=(schan,), img=input_, chdim=2)
                    d.addPairwiseEnergy(pairwise_energy, compat=compat)

                    pairwise_energy = create_pairwise_gaussian(sdims=(sdims, sdims), shape=(input_.shape[:2]))
                    d.addPairwiseEnergy(pairwise_energy, compat=compat)

                    # Run inference for 10 iterations
                    Q_unary = d.inference(iter)

                    # The Q is now the approximate posterior, we can get a MAP estimate using argmax.
                    map_soln_unary = np.argmax(Q_unary, axis=0)

                    # Unfortunately, the DenseCRF flattens everything, so get it back into picture form.
                    output_crf = map_soln_unary.reshape((output_crf.shape[1], output_crf.shape[2]))

                    ##
                    plt.imsave(os.path.join(dir_result_test_save, fileset['input']), input_.squeeze(), cmap=cmap)
                    plt.imsave(os.path.join(dir_result_test_save, fileset['output_th']), output_th.squeeze(), cmap=cmap)
                    plt.imsave(os.path.join(dir_result_test_save, fileset['output_crf']), output_crf.squeeze(), cmap=cmap)
                    plt.imsave(os.path.join(dir_result_test_save, fileset['label']), label_.squeeze(), cmap=cmap)

                    append_index(dir_result_test, fileset)

                print('TEST: %d/%d: LOSS: %.6f' % (i, num_batch_test, loss_G_cls.item()))
            print('TEST: AVERAGE LOSS: %.6f' % (np.mean(loss_G_cls_test)))


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
