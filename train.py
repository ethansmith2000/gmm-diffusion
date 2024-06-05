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
pool_probs=True
save_steps=500
resume_ckpt = 'diffusion.ckpt'
epochs = 300
# resume_ckpt = None

transform = transforms.Compose([transforms.Resize(size),
                                transforms.CenterCrop(size), 
                                transforms.RandomHorizontalFlip(0.5),transforms.ToTensor()])
train_dataset = torchvision.datasets.CIFAR10(root='./datasets/cifar10', train=True, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)


model = get_model(num_gaussians=num_gaussians).to(device)
scheduler = diffusers.DDIMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5",subfolder="scheduler")

optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=0.01, eps=1e-8)
nb_iter = 0

if resume_ckpt:
    state_dict = torch.load(resume_ckpt)
    nb_iter = state_dict['step']
    model.load_state_dict(state_dict['state_dict'])
    print('Loaded checkpoint')


def sample_mixprob(mixprob):
    b, num_g, c, h, w = mixprob.shape
    if pool_probs:
        mixprob = mixprob.flatten(2).mean(-1)
        mixprob = torch.nn.functional.softmax(mixprob, dim=-1)
        indices = torch.multinomial(mixprob, 1)
        indices = indices[:,:,None,None,None].repeat(1,1,c,h,w)
    else:
        mixprob = mixprob.permute(0, 2, 3, 4, 1).reshape(-1, num_g)
        probabilities = torch.nn.functional.softmax(mixprob, dim=-1)
        indices = torch.multinomial(probabilities, 1)
        indices = indices.view(b, c, h, w)[:,None,...]

    return indices

def loss_fn(pred, target):
    # channnels goes [mean1_r, mean1_g, mean1_b, mean2_r... logvar1_r, logvar1_g, logvar1_b, logvar2_r..., mixprob1_r, mixprob1_g, mixprob1_b, mixprob2_r...]
    means, logvar, mixprob = torch.chunk(pred, 3, dim=1)
    means, logvar, mixprob = map(lambda x: x.view(x.shape[0], num_gaussians, 3, x.shape[2], x.shape[3]).flatten(2), [means, logvar, mixprob])
    if pool_probs:
        mixprob = mixprob.mean(-1, keepdim=True)
    target = target.flatten(1)[:,None,:].repeat(1,num_gaussians,1)

    stds = (logvar.exp() + 1e-8).sqrt()
    diff = (target - means)
    mixprob = mixprob.softmax(-2)

    log_prob = -0.5 * (diff / stds) ** 2 - torch.log(stds) - 0.5 * torch.log(2 * torch.tensor(3.1415))
    log_probs = (log_prob + torch.log(mixprob + 1e-8)).logsumexp(-2)

    return -log_probs.mean()


@torch.no_grad()
def sample(model, scheduler, num_steps=50, batch_size=128):
    xt = torch.randn(batch_size, 3, size, size).to(device)
    scheduler.set_timesteps(num_steps)
    for t in range(num_steps):
        pred = model(xt, t)['sample']
        means, logvar, mixprob = torch.chunk(pred, 3, dim=1)
        means, logvar, mixprob = map(lambda x: x.view(x.shape[0], num_gaussians, 3, x.shape[2], x.shape[3]), [means, logvar, mixprob])
        indices = sample_mixprob(mixprob)
        means = means.gather(1, indices).squeeze(1)
        logvar = logvar.gather(1, indices).squeeze(1)
        pred = means + torch.randn_like(means) * (logvar.exp() + 1e-8).sqrt()
        xt = scheduler.step(pred, t, xt, return_dict=False)[0]

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
