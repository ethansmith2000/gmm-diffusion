import math
import torch
import torchvision
from torchvision import transforms
from diffusers import UNet2DModel
from torch.optim import AdamW, Adam
import diffusers
from tqdm import tqdm

def get_model(num_gaussians=3):
    # block_out_channels=(128, 128, 256, 256, 512, 512)
    block_out_channels=(128, 128, 256, 256, 512)
    # block_out_channels=(256, 256, 256, 512, 1024)
    down_block_types=( 
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D", 
        "DownBlock2D", 
        "DownBlock2D", 
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        # "DownBlock2D",
    )
    up_block_types=(
        # "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D", 
        "UpBlock2D", 
        "UpBlock2D", 
        "UpBlock2D"  
    )
    return UNet2DModel(block_out_channels=block_out_channels,out_channels=3 * 3 * num_gaussians, in_channels=3, up_block_types=up_block_types, down_block_types=down_block_types, add_attention=True)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
size = 32
num_gaussians=3
batch_size=256
lr = 0.4e-4
save_steps=500
resume_ckpt = 'diffusion.ckpt'
epochs = 300
# resume_ckpt = None
entropy_weight = 1e-3
diversity_weight = 5e-4

transform = transforms.Compose([transforms.Resize(size),
                                transforms.CenterCrop(size), 
                                transforms.RandomHorizontalFlip(0.5),transforms.ToTensor()])
train_dataset = torchvision.datasets.CIFAR10(root='./datasets/cifar10', train=True, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)


model = get_model(num_gaussians=num_gaussians).to(device)
scheduler = diffusers.DDIMScheduler.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    subfolder="scheduler",
)

optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=0.01, eps=1e-8)
nb_iter = 0

if resume_ckpt:
    state_dict = torch.load(resume_ckpt)
    nb_iter = state_dict['step']
    model.load_state_dict(state_dict['state_dict'])
    print('Loaded checkpoint')


def sample_mixprob(mix_logits):
    """
    Draw per-pixel component assignments using spatially-aware logits.
    """
    b, num_g, h, w = mix_logits.shape
    probs = torch.softmax(mix_logits.permute(0, 2, 3, 1).reshape(-1, num_g), dim=-1)
    indices = torch.multinomial(probs, 1).view(b, h, w)
    indices = indices[:, None, None, :, :].expand(-1, 1, 3, -1, -1)
    return indices


def reshape_mixture_tensors(pred):
    means, logvar, mixprob = torch.chunk(pred, 3, dim=1)
    b, _, h, w = means.shape
    means = means.view(b, num_gaussians, 3, h, w)
    logvar = logvar.view(b, num_gaussians, 3, h, w)
    mix_logits = mixprob.view(b, num_gaussians, 3, h, w).mean(dim=2)
    return means, logvar, mix_logits


def sample_x0_from_mixture(pred):
    means, logvar, mix_logits = reshape_mixture_tensors(pred)
    indices = sample_mixprob(mix_logits)
    means = means.gather(1, indices).squeeze(1)
    logvar = logvar.gather(1, indices).squeeze(1)
    std = (logvar.exp().clamp_min(1e-5)).sqrt()
    return means + torch.randn_like(means) * std


def convert_prediction_for_scheduler(x0_pred, xt, t, scheduler, prediction_type=None):
    """
    Convert x0 prediction into the representation expected by the scheduler.
    """
    prediction_type = prediction_type or scheduler.config.prediction_type
    if not isinstance(t, torch.Tensor):
        t = torch.tensor([t], device=xt.device, dtype=torch.long)
    else:
        t = t.to(device=xt.device, dtype=torch.long)
    if t.dim() == 0:
        t = t.unsqueeze(0)
    if t.shape[0] == 1 and xt.shape[0] > 1:
        t = t.expand(xt.shape[0])
    elif t.shape[0] != xt.shape[0]:
        raise ValueError("Timestep shape must match batch size or be scalar.")

    alphas_cumprod = scheduler.alphas_cumprod.to(xt.device)
    alpha_t = alphas_cumprod.index_select(0, t).view(-1, 1, 1, 1)
    sqrt_alpha = alpha_t.sqrt()
    sqrt_one_minus_alpha = (1 - alpha_t).clamp_min(1e-12).sqrt()
    epsilon = (xt - sqrt_alpha * x0_pred) / sqrt_one_minus_alpha

    if prediction_type == "epsilon":
        return epsilon
    if prediction_type == "v_prediction":
        return sqrt_alpha * epsilon - sqrt_one_minus_alpha * x0_pred
    if prediction_type == "sample":
        return x0_pred
    raise ValueError(f"Unsupported prediction_type: {prediction_type}")

def loss_fn(pred, target):
    """
    Negative log-likelihood for per-pixel Gaussian mixtures with diversity/entropy regularizers.
    """
    means, logvar, mix_logits = reshape_mixture_tensors(pred)
    b = means.shape[0]

    log_mix = torch.nn.functional.log_softmax(mix_logits, dim=1)
    mix_probs = log_mix.exp()

    target = target[:, None, :, :, :]
    var = logvar.exp().clamp_min(1e-5)
    log_component = -0.5 * (((target - means) ** 2) / var + logvar + math.log(2 * math.pi))
    log_component = log_component.sum(dim=2)
    recon_loss = -torch.logsumexp(log_mix + log_component, dim=1).mean()

    entropy = -(mix_probs * log_mix).sum(dim=1).mean()
    entropy_reg = -entropy_weight * entropy

    comp_means = means.view(b, num_gaussians, 3, -1).mean(dim=-1)
    pairwise_sq = (comp_means[:, :, None, :] - comp_means[:, None, :, :]).pow(2).sum(dim=-1)
    if num_gaussians > 1:
        mask = torch.eye(num_gaussians, device=pred.device).bool()
        pairwise_sq = pairwise_sq.masked_select(~mask)
        diversity_loss = diversity_weight * torch.exp(-pairwise_sq).mean()
    else:
        diversity_loss = pred.new_tensor(0.0)

    return recon_loss + entropy_reg + diversity_loss


@torch.no_grad()
def sample(model, scheduler, num_steps=50, batch_size=128):
    xt = torch.randn(batch_size, 3, size, size).to(device)
    scheduler.set_timesteps(num_steps)
    prediction_type = scheduler.config.prediction_type
    for t in scheduler.timesteps:
        pred = model(xt, t)['sample']
        x0_pred = sample_x0_from_mixture(pred)
        scheduler_pred = convert_prediction_for_scheduler(x0_pred, xt, t, scheduler, prediction_type)
        xt = scheduler.step(scheduler_pred, t, xt, return_dict=False)[0]

    return xt


print('Start training')
pbar = tqdm(total=epochs * len(dataloader), initial=nb_iter)
# torch.set_float32_matmul_precision("medium")
# model = torch.compile(model)
for current_epoch in range(100):
    for i, data in enumerate(dataloader):
        # with torch.amp.autocast(enabled=True, device_type='cuda', dtype=torch.bfloat16):
        x0 = (data[0].to(device)*2)-1
        noise = torch.randn_like(x0)
        t = torch.randint(0,scheduler.config.num_train_timesteps,(x0.shape[0],),device=device,).long()
        xt = scheduler.add_noise(x0, noise, t).to(x0.dtype)
        
        pred = model(xt, t)['sample']

        loss = loss_fn(pred, x0)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        nb_iter += 1
        pbar.update(1)
        pbar.set_description(f'loss: {loss.item()}')

        if nb_iter % save_steps == 0:
            with torch.no_grad():
                print(f'Save export {nb_iter}')
                outputs = (sample(model, scheduler, num_steps=128, batch_size=128) * 0.5) + 0.5
                torchvision.utils.save_image(outputs, f'export_{str(nb_iter).zfill(8)}.png')
                state_dict = {
                    "state_dict": model.state_dict(),
                    "step": nb_iter,
                }
                torch.save(state_dict, f'diffusion.ckpt')
