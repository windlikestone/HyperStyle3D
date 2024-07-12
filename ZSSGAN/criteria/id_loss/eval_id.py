from PIL import Image
import torchvision.transforms as transforms
import torch
from tqdm import tqdm
# from lpips import LPIPS
# from id_loss import IDLoss
from id_loss import IDLoss
# from pytorch_msssim import ms_ssim, ssim


class Eval:
    def __init__(self):
        self.l2 = torch.nn.MSELoss(reduction='mean').cuda()
        # self.lpips = LPIPS(net='alex').cuda().eval()
        # self.id_loss = IDLoss().cuda().eval()
        self.id_ckpt_path = '/home/chenzhuo/workspace/cartoonGAN/model_zoo/model_ir_se50.pth'
        self.id_loss = IDLoss(self.id_ckpt_path).cuda().eval()

        self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
            )

    # def eval_mse(self, gt_root, fake_root, eval_num):
    #     error = 0.0
    #     count = 0

    #     for i in tqdm(range(eval_num)):
    #         try:
    #             gt_path = gt_root + str(i) + '.jpg'
    #             gt_img = Image.open(gt_path).convert('RGB')
    #             gt_img = self.transform(gt_img).cuda()

    #             fake_path = fake_root + str(i) + '.png'
    #             fake_img = Image.open(fake_path).convert('RGB')
    #             fake_img = self.transform(fake_img).cuda()

    #             error += self.l2(gt_img, fake_img)
    #             count += 1
    #         except:
    #             continue

    #     error /= count
    #     print(error)

    # def eval_lpips(self, gt_root, fake_root, eval_num):
    #     error = 0.0
    #     count = 0

    #     for i in tqdm(range(eval_num)):
    #         try:
    #             gt_path = gt_root + str(i) + '.jpg'
    #             gt_img = Image.open(gt_path).convert('RGB')
    #             gt_img = self.transform(gt_img).cuda()

    #             fake_path = fake_root + str(i) + '.png'
    #             fake_img = Image.open(fake_path).convert('RGB')
    #             fake_img = self.transform(fake_img).cuda()

    #             error += torch.squeeze(self.lpips(gt_img, fake_img))
    #             count += 1
    #         except:
    #             continue

    #     error /= count
    #     print(error)

    def eval_id(self, gt_root, fake_root, eval_num):
        error = []

        for i in tqdm(range(eval_num)):
            
            gt_path = gt_root + str(i).zfill(7) + '_azim0_elev0.png'
            gt_img = Image.open(gt_path).convert('RGB')
            gt_img = self.transform(gt_img).cuda()
            # print(gt_img)

            # fake_path = fake_root + str(i) + '.png'
            fake_path = fake_root + str(i).zfill(7) + '_azim25_elev0.png'
            fake_img = Image.open(fake_path).convert('RGB')
            fake_img = self.transform(fake_img).cuda()
            # print(fake_img)
            err = 1 - self.id_loss.forward(gt_img.unsqueeze(0), fake_img.unsqueeze(0))
            if err < 0.1: continue
            error.append(err)
            
        

        print("Id SIMILARITY", sum(error) / len(error))

    # def eval_ssim(self, gt_root, fake_root, eval_num):
    #     error = 0.0
    #     count = 0

    #     for i in tqdm(range(eval_num)):
    #         try:
    #             gt_path = gt_root + str(i) + '.jpg'
    #             gt_img = Image.open(gt_path).convert('RGB')
    #             gt_img = self.transform(gt_img)

    #             fake_path = fake_root + str(i) + '.png'
    #             fake_img = Image.open(fake_path).convert('RGB')
    #             fake_img = self.transform(fake_img)

    #             error += ms_ssim(gt_img.unsqueeze(0), fake_img.unsqueeze(0), data_range=1)
    #             count += 1
    #         except:
    #             continue

    #     error /= count
    #     print(error)


if __name__ == '__main__':
    eval_model = Eval()
    """
    gt_root = '/2t/datasets/EG3D_CelebA/final_crops/'
    fake_root = 'inversion/CelebA_inv/'
    pti_root = 'inversion/CelebA_inv_none/'

    length = 250

    eval_model.eval_ssim(gt_root, fake_root, length)
    eval_model.eval_ssim(gt_root, pti_root, length)

    eval_model.eval_mse(gt_root, fake_root, length)
    eval_model.eval_mse(gt_root, pti_root, length)

    eval_model.eval_lpips(gt_root, fake_root, length)
    eval_model.eval_lpips(gt_root, pti_root, length)

    eval_model.eval_id(gt_root, fake_root, length)
    eval_model.eval_id(gt_root, pti_root, length)
    """

    # eval_model.eval_id('/2t/datasets/CelebAMask-HQ/CelebA-HQ-img/', '/home/lyx0208/Desktop/face/MegaFS/output/megafs/', 100)
    eval_model.eval_id('/home/chenzhuo/workspace/StyleGAN3D/eval_3D_consistency_file/img/beard_fat_0/', '/home/chenzhuo/workspace/StyleGAN3D/eval_3D_consistency_file/img/beard_fat_view/', 50)
    # eval_model.eval_id('/2t/datasets/EG3D_CelebA/final_crops/', '/home/lyx0208/Desktop/face/eg3d/eg3d/results/fslsd/', 1000)
    # eval_model.eval_id('/2t/datasets/EG3D_CelebA/final_crops/', '/home/lyx0208/Desktop/face/eg3d/eg3d/results/infoswap/', 1000)
    # eval_model.eval_id('/2t/datasets/EG3D_CelebA/final_crops/', '/home/lyx0208/Desktop/face/eg3d/eg3d/results/simswap/', 1000)
