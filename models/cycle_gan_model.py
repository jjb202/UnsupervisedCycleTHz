import torch
import itertools
import torch.nn.functional as F
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import numpy as np
import cv2
# from models.dcd_model import Butterworth
# from models.focal_frequency_loss import FocalFrequencyLoss as FFL

class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_L1', type=float, default=1.0, help='weight for L1 loss')
            parser.add_argument('--lambda_feat', type=float, default=0.2, help='weight for feature matching loss')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'D_B', 'G_B', 'cycle_B']
        # self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        # if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
        #     visual_names_A.append('idt_B')
        #     visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionL1 = torch.nn.L1Loss().to(self.device)
            self.perceptual = networks.perceptual().to(self.device)

            # self.criterionFrequency = Butterworth().to(self.device)
            # define focal frequency loss
            # self.criterion_freq = FFL(loss_weight=1.0,
            #                           alpha=1.0,
            #                           patch_factor=1,
            #                           ave_spectrum=True,
            #                           log_matrix=True,
            #                           batch_matrix=True).to(self.device)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        # mixup method
        lam = np.random.beta(1,1)
        index = torch.randperm(self.real_A.size(0))
        self.real_A = lam * self.real_A + (1 - lam) * self.real_A[index, :]

        lam = np.random.beta(1, 1)
        index2 = torch.randperm(self.real_B.size(0))
        self.real_B = lam * self.real_B + (1 - lam) * self.real_B[index2, :]

        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx - cut_w // 2, 0, W)
        bby2 = np.clip(cy - cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.isTrain:
            self.fake_B = self.netG_A(self.real_A)  # G_A(A)
            self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))

            self.fake_A = self.netG_B(self.real_B)  # G_B(B)
            self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))
        else:
            h,w = self.real_A.shape[2], self.real_A.shape[3]
            H,W = ((h+256)//256)*256, ((w+256)//256)*256
            padh = H-h if h%8!=0 else 0
            padw = W-w if w%8!=0 else 0
            self.real_A = F.pad(self.real_A, (0,padw,0,padh), 'reflect')

            h,w = self.real_B.shape[2], self.real_B.shape[3]
            H,W = ((h+256)//256)*256, ((w+256)//256)*256
            padh = H-h if h%8!=0 else 0
            padw = W-w if w%8!=0 else 0
            self.real_B = F.pad(self.real_B, (0,padw,0,padh), 'reflect')

            self.fake_B = self.netG_A(self.real_A)  # G_A(A)
            self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))

            self.fake_A = self.netG_B(self.real_B)  # G_B(B)
            self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))
            self.fake_B = self.fake_B[:, :, :h, :w]
            self.rec_A = self.rec_A[:, :, :h, :w]
            self.fake_A = self.fake_A[:, :, :h, :w]
            self.rec_B = self.rec_B[:, :, :h, :w]
            self.real_A = self.real_A[:, :, :h, :w]
            self.real_B = self.real_B[:, :, :h, :w]

    # def fft(self, input, D):
    #     # 傅里叶变换
    #     f1 = np.fft.fft2(input.cpu())
    #     # 使用np.fft.fftshift()函数实现平移，让直流分量输出图像的重心
    #     f1_shift = np.fft.fftshift(f1)
    #     # 实现理想低通滤波器
    #     rows, cols = input.shape
    #     crow, ccol = int(rows / 2), int(cols / 2)  # 计算频谱中心
    #     mask = np.zeros((rows, cols), dtype='uint8')  # 生成rows行，从cols列的矩阵，数据格式为uint8
    #     # 将距离频谱中心距离小于D的低通信息部分设置为1，属于低通滤波
    #     for i in range(rows):
    #         for j in range(cols):
    #             if np.sqrt(i * i + j * j) <= D:
    #                 mask[crow - D:crow + D, ccol - D:ccol + D] = 1
    #     f1_shift = f1_shift * mask
    #     # 傅里叶逆变换
    #     f_ishift = np.fft.ifftshift(f1_shift)
    #     img_back = np.fft.ifft2(f_ishift)
    #     img_back = np.abs(img_back)
    #     img_back = (img_back - np.amin(img_back)) / (np.amax(img_back) - np.amin(img_back))
    #
    #     return img_back.cuda()

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        self.fake_B_red = self.fake_B[:, 0:1, :, :]
        self.fake_B_green = self.fake_B[:, 1:2, :, :]
        self.fake_B_blue = self.fake_B[:, 2:3, :, :]

        self.real_B_red = self.real_B[:, 0:1, :, :]
        self.real_B_green = self.real_B[:, 1:2, :, :]
        self.real_B_blue = self.real_B[:, 2:3, :, :]

        self.fake_A_red = self.fake_A[:, 0:1, :, :]
        self.fake_A_green = self.fake_A[:, 1:2, :, :]
        self.fake_A_blue = self.fake_A[:, 2:3, :, :]

        self.real_A_red = self.real_A[:, 0:1, :, :]
        self.real_A_green = self.real_A[:, 1:2, :, :]
        self.real_A_blue = self.real_A[:, 2:3, :, :]

        self.loss_G_L1 = (self.criterionL1(self.fake_B_red, self.real_B_red) + self.criterionL1(self.fake_B_green,
                                                                                                self.real_B_green) + self.criterionL1(
            self.fake_B_blue, self.real_B_blue)) * self.opt.lambda_L1 + (self.criterionL1(self.fake_A_red, self.real_A_red) + self.criterionL1(
                                 self.fake_A_green, self.real_A_green) + self.criterionL1(self.fake_A_blue,
                                                                                          self.real_A_blue)) * self.opt.lambda_L1
        self.percept = self.perceptual(self.fake_B, self.fake_A)
        # self.loss_G_Frequency = self.criterionFrequency(self.rec_B,
        #                                                 self.real_B) * self.opt.lambda_feat + self.criterionFrequency(
        #     self.rec_A, self.real_A) * self.opt.lambda_feat
        #
        # self.criterion_foce = self.criterion_freq(self.rec_B,
        #                                                 self.real_B) * self.opt.lambda_feat + self.criterionFrequency(
        #     self.rec_A, self.real_A) * self.opt.lambda_feat

        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_G_L1 + self.percept
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights