from utils.utils import *
import torch

from torchvision import transforms

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


esd_path = 'models/esd-vangogh_from_vangogh-xattn_1-epochs_200.pt'
pl_path = 'models/pl-vangogh_from_vangogh-xattn_1-epochs_200.pt'
new_pl_path = 'models/Pos_new_pl-vangogh_from_vangogh-xattn_1-epochs_200.pt'
train_method = 'xattn' ## REMEMBER: please use the same train_method you used for training (it is present in the saved name)

diffuser = StableDiffuser(scheduler='DDIM').to('cuda')

finetuner = FineTunedModel(diffuser, train_method=train_method)
finetuner.load_state_dict(torch.load(esd_path))

import random
seed = random.randint(0,2**15)
seed = 41
print(seed)
prompts = ["Van Gogh", "unicorn draw with the Van Gogh style", "unicorn"]
for prompt in prompts:
    # images = diffuser(prompt,
    #          img_size=512,
    #          n_steps=50,
    #          n_imgs=1,
    #          generator=torch.Generator().manual_seed(seed),
    #          guidance_scale=7.5
    #          )[0][0]
    # images = transforms.ToTensor()(images)
    # plt.imshow(images.permute(1,2,0).cpu().numpy())
    # plt.axis('off')
    # plt.savefig('figs/ori_{}.png'.format(prompt.replace(' ','_')))
# 
# 
# 
    # with finetuner:
    #     images = diffuser(prompt,
    #              img_size=512,
    #              n_steps=50,
    #              n_imgs=1,
    #              generator=torch.Generator().manual_seed(seed),
    #              guidance_scale=7.5
    #              )[0][0]
    # images = transforms.ToTensor()(images)
    # plt.imshow(images.permute(1,2,0).cpu().numpy())
    # plt.axis('off')
    # plt.savefig('figs/est_{}.png'.format(prompt.replace(' ','_')))
# 
# 
    # n_opt_prompts = 3
    # masked_prompt_embedding = None
# 
    # # torch load
    # masked_prompt_embedding = torch.load(pl_path)
# 
    # images = diffuser(prompt,
    #          img_size=512,
    #          n_steps=50,
    #          n_imgs=1,
    #          generator=torch.Generator().manual_seed(seed),
    #          guidance_scale=7.5,
    #          n_opt_prompts=n_opt_prompts,
    #         masked_prompt_embedding=masked_prompt_embedding
    #          )[0][0]
    # images = transforms.ToTensor()(images)
    # plt.imshow(images.permute(1,2,0).cpu().numpy())
    # plt.axis('off')
    # plt.savefig('figs/pl_{}.png'.format(prompt.replace(' ','_')))
    # 
    # 
    n_opt_prompts = 3
    masked_prompt_embedding = None

    # torch load
    masked_prompt_embedding = torch.load(new_pl_path)
    # import pdb;pdb.set_trace()
    images = diffuser(prompt,
             img_size=512,
             n_steps=50,
             n_imgs=1,
             generator=torch.Generator().manual_seed(seed),
             guidance_scale=7.5,
             n_opt_prompts=n_opt_prompts,
            masked_prompt_embedding=masked_prompt_embedding
             )[0][0]
    images = transforms.ToTensor()(images)
    plt.imshow(images.permute(1,2,0).cpu().numpy())
    plt.axis('off')
    plt.savefig('figs/Posnew_pl_{}.png'.format(prompt.replace(' ','_')))