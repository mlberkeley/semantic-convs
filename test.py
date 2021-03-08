from easydict import EasyDict
from scae.modules.constellation_ae import ConstellationCapsule
from scae.modules.attention import SetTransformer
from scae.modules.capsule import CapsuleLayer
from scae.models.ccae import CCAE
from scae.data.constellation import create_constellation
from torch.utils.data import DataLoader
import wandb
import torch

logging = True

if logging:
    wandb.init(project="StackedCapsuleAutoEncoders")
    wandb.run.name = "capsule routing intuition"
    wandb.run.save()

args = EasyDict(
    batch_size=1,
    model="ccae",
    dataset="constellation",
    ccae=EasyDict(
        lr=0.01,
        lr_decay=0.998,
        weight_decay=0.0
    )
)


# data_gen = create_constellation(
#     batch_size=args.batch_size,
#     shuffle_corners=True,
#     gaussian_noise=.0,
#     drop_prob=0.5,
#     which_patterns=[[0], [1], [0]],
#     rotation_percent=180 / 360.,
#     max_scale=3.,
#     min_scale=3.,
#     use_scale_schedule=False,
#     schedule_steps=0,
# )


def const_to_wandb_scatter(const_batch):
    """Takes the first constellation group in a batch and scatter plots.
    """
    constellation = const_batch.corners[0]
    scatter = []
    for i in range(constellation.shape[0]):
        scatter.append((const_batch.pattern_id[0][i]+1,
                        constellation[i][0], constellation[i][1]))
        table = wandb.Table(data=scatter, columns=["pattern_id", "x", "y"])
        wandb.log({'one': wandb.plot.scatter(
            table, "x", "y")})


def capsule_vote_to_wandb_scatter(capsule_vote_batch, epoch):
    """Takes the first set of capsule votes in a batch and scatter plots.
    """
    votes = []
    for i in range(12):
        x, y = capsule_vote_batch[0][i]
        votes.append(((i // 4)+1, x, y))
        table = wandb.Table(data=votes, columns=["capsule", "x", "y"])
        wandb.log({f'capsule_vote {epoch}': wandb.plot.scatter(
            table, "x", "y")})


# wandb.log({"presence": data_gen.presence})
# wandb.log({"pattern_presence": data_gen.pattern_presence})
# wandb.log({"pattern_id": data_gen.pattern_id})
encoder = SetTransformer(2)
decoder = CapsuleLayer(
    input_dims=32, n_caps=3, n_caps_dims=2, n_votes=4)
model = CCAE(encoder, decoder, args)
opt = torch.optim.Adam(model.parameters(), lr=0.001)


train_data = [create_constellation(
    batch_size=1,
    shuffle_corners=True,
    gaussian_noise=.0,
    drop_prob=0.5,
    which_patterns=[[0], [1], [0]],
    rotation_percent=180 / 360.,
    max_scale=3.,
    min_scale=3.,
    use_scale_schedule=False,
    schedule_steps=0,
) for i in range(1)]

const_to_wandb_scatter(train_data[0])

for i in range(20):
    for data in train_data:
        patterns = torch.from_numpy(data.corners)
        out = model.forward(patterns)

        opt.zero_grad()
        loss = out.log_prob + out.sparsity_loss
        presence = out.caps_presence_prob
        print(i)
        print(loss)
        print(out.vote)
        # print(presence)

        # print(i, torch.bincount(torch.flatten(caps_num)))
        loss.backward()
        opt.step()

        if i % 5 == 0:
            capsule_vote_to_wandb_scatter(out.vote, i)

        if logging:
            wandb.log({'epoch': i})
            wandb.log({'log_prob': out.log_prob})
            wandb.log({'sparsity_loss': out.sparsity_loss})
            wandb.log({'mean caps_presence_prob':
                       torch.mean(out.caps_presence_prob)})
            wandb.log({'mean_scale': out.mean_scale})
