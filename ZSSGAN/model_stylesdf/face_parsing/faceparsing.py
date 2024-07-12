import torch
import torchvision.transforms as transforms
from model_stylesdf.face_parsing.parsingmodel import BiSeNet

class FaceParsing(torch.nn.Module):
    def __init__(self, device):
        super(FaceParsing, self).__init__()
        self.device = device
        self.parsing_ckpt = "//home/chenzhuo/workspace/3DAnimationGAN/model_zoo/79999_iter.pth"
        self.parsing_net = BiSeNet(n_classes=19).to(self.device)
        self.parsing_net.load_state_dict(torch.load(self.parsing_ckpt))
        self.parsing_net.eval()
        self.atts = {1: 'skin', 2: 'l_brow', 3: 'r_brow', 4: 'l_eye', 5: 'r_eye', 6: 'eye_g', 7: 'l_ear', 8: 'r_ear', 9: 'ear_r',
                    10: 'nose', 11: 'mouth', 12: 'u_lip', 13: 'l_lip', 14: 'neck', 15: 'neck_l', 16: 'cloth', 17: 'hair', 18: 'hat'}
    
    def region_mask(self, im, parsing_anno, stride, index=17, reverse_mask=False):

        vis_parsing_anno = parsing_anno.type(torch.uint8)
        
        zeros = torch.zeros_like(vis_parsing_anno)
        ones = torch.ones_like(vis_parsing_anno)
        hair_mask = torch.where(vis_parsing_anno == index, zeros, ones) # 17 -> hair
        if reverse_mask:
            hair_mask = torch.where(vis_parsing_anno == index, ones, zeros) # 17 -> hair

        return hair_mask

    def face_mask(self, frozen_img, index=17, reverse_mask=True):

        to_tensor = transforms.Compose([
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        f_img = to_tensor(frozen_img)
        f_out = self.parsing_net(f_img)[0]
        f_parsing = f_out.argmax(1)

        f_mask = self.region_mask(f_img, f_parsing, stride=1, index=index, reverse_mask=reverse_mask).unsqueeze(1).repeat(1, 3, 1, 1)

        return f_mask