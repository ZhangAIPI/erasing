from PIL import Image
from matplotlib import pyplot as plt
import textwrap
import argparse
import torch
import copy
import os
import re
import numpy as np
from diffusers import AutoencoderKL, UNet2DConditionModel
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPFeatureExtractor
from diffusers.schedulers import EulerAncestralDiscreteScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_lms_discrete import LMSDiscreteScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from utils.utils import *



def train(erase_concept, erase_from, train_method, iterations, negative_guidance, lr, save_path, device):
  
    nsteps = 50

    diffuser = StableDiffuser(scheduler='DDIM').to(device)
    diffuser.train()

    finetuner = FineTunedModel(diffuser, train_method=train_method)

    # optimizer = torch.optim.Adam(finetuner.parameters(), lr=lr)
    criteria = torch.nn.MSELoss()

    pbar = tqdm(range(iterations))
    erase_concept = erase_concept.split(',')
    erase_concept = [a.strip() for a in erase_concept]
    
    erase_from = erase_from.split(',')
    erase_from = [a.strip() for a in erase_from]
    
    
    if len(erase_from)!=len(erase_concept):
        if len(erase_from) == 1:
            c = erase_from[0]
            erase_from = [c for _ in erase_concept]
        else:
            print(erase_from, erase_concept)
            raise Exception("Erase from concepts length need to match erase concepts length")
            
    erase_concept_ = []
    for e, f in zip(erase_concept, erase_from):
        erase_concept_.append([e,f])
    
    
    
    erase_concept = erase_concept_
    
    
    
    print(erase_concept)

    torch.cuda.empty_cache()
    n_opt_prompts = 3
    
    loss_dynamics = []

    for index in range(len(erase_concept)):
        
        erase_concept_sampled = erase_concept[index]
        neutral_text_embeddings,to_optimize_embeddings, neutral_opt_token_stard_idx, neutral_opt_token_end_idx = diffuser.get_text_embeddings_with_OPTprompts([''],n_imgs=1)
        # optimize the to_optimize_embeddings as the variable to optimize
        to_optimize_embeddings = torch.nn.Parameter(to_optimize_embeddings.clone().detach())
        to_optimize_embeddings = to_optimize_embeddings/ to_optimize_embeddings.norm(dim=-1, keepdim=True, p=2)
        to_optimize_embeddings = to_optimize_embeddings.requires_grad_(True)
        # optimizer = torch.optim.Adam([to_optimize_embeddings], lr=lr)
        # neutral_text_embeddings[:,neutral_opt_token_stard_idx:neutral_opt_token_end_idx] = to_optimize_embeddings.clone()
        neutral_text_embeddings = diffuser.get_text_embeddings([''],n_imgs=1)
        positive_text_embeddings, positive_opt_token_stard_idx, positive_opt_token_end_idx = diffuser.get_text_embeddings_with_PostOPTprompts([erase_concept_sampled[0]],n_imgs=1,n_opt_prompts=n_opt_prompts, masked_prompt_embedding=to_optimize_embeddings)
        # target_text_embeddings, target_opt_token_stard_idx, target_opt_token_end_idx = diffuser.get_text_embeddings_with_PostOPTprompts([erase_concept_sampled[1]],n_imgs=1,n_opt_prompts=n_opt_prompts, masked_prompt_embedding=to_optimize_embeddings)
        target_text_embeddings = diffuser.get_text_embeddings([erase_concept_sampled[1]],n_imgs=1)
        diffuser.set_scheduler_timesteps(nsteps)
        # optimizer.zero_grad()
        for i in pbar:
            iteration = torch.randint(1, nsteps - 1, (1,)).item()
            # print("iteration", iteration)
            diffuser.set_scheduler_timesteps(nsteps)
            latents = diffuser.get_initial_latents(1, 512, 1)
            with finetuner:
                latents_steps, _ = diffuser.diffusion(
                    latents,
                    positive_text_embeddings,
                    start_iteration=0,
                    end_iteration=iteration,
                    guidance_scale=3, 
                    show_progress=False
                ) 
            diffuser.set_scheduler_timesteps(1000)
            iteration = int(iteration / nsteps * 1000)
            positive_latents = diffuser.predict_noise(iteration, latents_steps[0], positive_text_embeddings, guidance_scale=1)
            neutral_latents = diffuser.predict_noise(iteration, latents_steps[0], neutral_text_embeddings, guidance_scale=1)
            target_latents = diffuser.predict_noise(iteration, latents_steps[0], target_text_embeddings, guidance_scale=1)
            if erase_concept_sampled[0] == erase_concept_sampled[1]:
                target_latents = neutral_latents.clone().detach()
            with finetuner:
                negative_latents = diffuser.predict_noise(iteration, latents_steps[0], target_text_embeddings, guidance_scale=1)
            # positive_latents.requires_grad = False
            # neutral_latents.requires_grad = False

            loss = criteria(negative_latents, target_latents - (negative_guidance*(positive_latents - neutral_latents))) 
            loss_dynamics.append(loss.cpu().item())
            # loss.backward(retain_graph=True)
            # import pdb;pdb.set_trace()
            # optimizer.step()
            grad = torch.autograd.grad(loss, to_optimize_embeddings, retain_graph=False, create_graph=False)[0]
            with torch.no_grad():
                to_optimize_embeddings = to_optimize_embeddings - lr * grad
                # to_optimize_embeddings = to_optimize_embeddings/ to_optimize_embeddings.norm(dim=-1, keepdim=True, p=2)
            
            to_optimize_embeddings.requires_grad_(True)
            
            positive_text_embeddings[:,positive_opt_token_stard_idx:positive_opt_token_end_idx] = to_optimize_embeddings.clone()
            # target_text_embeddings[:,target_opt_token_stard_idx:target_opt_token_end_idx] = to_optimize_embeddings.clone()
            # neutral_text_embeddings[:,neutral_opt_token_stard_idx:neutral_opt_token_end_idx] = to_optimize_embeddings.clone()
            

    torch.save(to_optimize_embeddings.detach().cpu(), save_path)

    del diffuser, loss,  finetuner, negative_latents, neutral_latents, positive_latents, latents_steps, latents

    torch.cuda.empty_cache()
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.plot(loss_dynamics)
    plt.savefig(f'{save_path.replace(".pt","")}_loss.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'TrainESD',
                    description = 'Finetuning stable diffusion to erase the concepts')
    parser.add_argument('--erase_concept', help='concept to erase', type=str, required=True)
    parser.add_argument('--erase_from', help='target concept to erase from', type=str, required=False, default = None)
    parser.add_argument('--train_method', help='Type of method (xattn, noxattn, full, xattn-strict', type=str, required=True)
    parser.add_argument('--iterations', help='Number of iterations', type=int, default=500)
    parser.add_argument('--lr', help='Learning rate', type=float, default=1e-3)
    parser.add_argument('--negative_guidance', help='Negative guidance value', type=float, required=False, default=1)
    parser.add_argument('--save_path', help='Path to save model', type=str, default='models/')
    parser.add_argument('--device', help='cuda device to train on', type=str, required=False, default='cuda:0')

    args = parser.parse_args()
    
    prompt = args.erase_concept #'car'
    erase_concept = args.erase_concept
    erase_from = args.erase_from
    if erase_from is None:
        erase_from = erase_concept
    train_method = args.train_method #'noxattn'
    iterations = args.iterations #200
    negative_guidance = args.negative_guidance #1
    lr = args.lr #1e-5
    name = f"large_Pos_new_pl-{erase_concept.lower().replace(' ','').replace(',','')}_from_{erase_from.lower().replace(' ','').replace(',','')}-{train_method}_{negative_guidance}-epochs_{iterations}"
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok = True)
    save_path = f'{args.save_path}/{name}.pt'
    device = args.device
    train(erase_concept=erase_concept, erase_from=erase_from, train_method=train_method, iterations=iterations, negative_guidance=negative_guidance, lr=lr, save_path=save_path, device=device)
