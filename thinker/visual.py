import os
from thinker.util import __version__
from thinker.util import __project__

import cv2
import argparse
import numpy as np
import copy
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
import torch
import torch.nn.functional as F
import textwrap
from thinker.main import Env
from thinker.self_play import init_env_out, create_env_out
from thinker.actor_net import ActorNet
import thinker.util as util
import gym

def plot_gym_env_out(x, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(
        torch.swapaxes(torch.swapaxes(x[0, -3:].cpu(), 0, 2), 0, 1),
        interpolation="nearest",
        aspect="auto",
    )
    if title is not None:
        ax.set_title(title)


def plot_multi_gym_env_out(xs, titles=None, col_n=5):
    size_n = 6
    row_n = (len(xs) + (col_n - 1)) // col_n

    fig, axs = plt.subplots(row_n, col_n, figsize=(col_n * size_n, row_n * size_n))
    if len(axs.shape) == 1:
        axs = axs[np.newaxis, :]
    m = 0
    for y in range(row_n):
        for x in range(col_n):
            if m >= len(xs):
                axs[y][x].set_axis_off()
            else:
                axs[y][x].imshow(np.transpose(xs[m][-3:], axes=(1, 2, 0)) / 255)
                axs[y][x].set_title(
                    "rollout %d" % (m + 1) if titles is None else titles[m]
                )
            m += 1
    plt.tight_layout()
    return fig


def plot_policies(logits, labels, action_meanings, ax=None, title="Real policy prob"):
    if ax is None:
        fig, ax = plt.subplots()
    probs = []
    for logit, k in zip(logits, labels):
        if k != "action":
            probs.append(torch.softmax(logit, dim=-1).detach().cpu().numpy())
        else:
            probs.append(logit.detach().cpu().numpy())

    ax.set_title(title)
    xs = np.arange(len(probs[0]))
    for n, (prob, label) in enumerate(zip(probs, labels)):
        ax.bar(xs + 0.1 * (n - len(logits) // 2), prob, width=0.1, label=label)
    ax.xaxis.set_major_locator(mticker.FixedLocator(np.arange(len(probs[0]))))
    ax.set_xticklabels(action_meanings, rotation=90)
    plt.subplots_adjust(bottom=0.2)
    ax.set_ylim(0, 1)
    ax.legend()


def plot_base_policies(logits, action_meanings, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    prob = torch.softmax(logits, dim=-1).detach().cpu().numpy()
    rec_t, num_actions = logits.shape
    xs = np.arange(rec_t)
    labels = action_meanings
    for i in range(num_actions):
        c = ax.bar(
            xs + 0.8 * (i / num_actions),
            prob[:, i],
            width=0.8 / (num_actions),
            label=labels[i],
        )
        color = c.patches[0].get_facecolor()
        color = color[:3] + (color[3] * 0.5,)
        ax.bar(
            xs + 0.8 * (i / num_actions),
            prob[:, i],
            width=0.8 / (num_actions),
            color=color,
        )
    ax.legend()
    ax.set_ylim(0, 1)
    ax.set_title("Model policy prob")


def plot_im_policies(
    pri_logits,
    reset_logits,
    pri,
    reset,
    action_meanings,
    one_hot=True,
    reset_ind=0,
    ax=None,
):
    if ax is None:
        fig, ax = plt.subplots()

    rec_t, num_actions = pri_logits.shape
    num_actions += 1
    rec_t -= 1

    im_prob = torch.softmax(pri_logits, dim=-1).detach().cpu().numpy()
    reset_prob = (
        torch.softmax(reset_logits, dim=-1)[:, [reset_ind]]
        .detach()
        .cpu()
        .numpy()
    )
    full_prob = np.concatenate([im_prob, reset_prob], axis=-1)

    if not one_hot:
        pri = F.one_hot(pri, num_actions - 1)
    pri = pri.detach().cpu().numpy()
    reset = reset.unsqueeze(-1).detach().cpu().numpy()
    full_action = np.concatenate([pri, reset], axis=-1)

    xs = np.arange(rec_t + 1)
    labels = action_meanings.copy()
    labels.append("RESET")

    for i in range(num_actions):
        c = ax.bar(
            xs + 0.8 * (i / num_actions),
            full_prob[:, i],
            width=0.8 / (num_actions),
            label=labels[i],
        )
        color = c.patches[0].get_facecolor()
        color = color[:3] + (color[3] * 0.5,)
        ax.bar(
            xs + 0.8 * (i / num_actions),
            full_action[:, i],
            width=0.8 / (num_actions),
            color=color,
        )
    ax.legend()
    ax.set_ylim(0, 1)
    ax.set_title("Imagainary policy prob")


def plot_qn_sa(q_s_a, n_s_a, action_meanings, max_q_s_a=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    xs = np.arange(len(q_s_a))

    ax.bar(xs - 0.3, q_s_a.cpu(), color="g", width=0.3, label="q_s_a")
    ax_n = ax.twinx()
    if max_q_s_a is not None:
        ax.bar(xs, max_q_s_a.cpu(), color="r", width=0.3, label="max_q_s_a")
    ax_n.bar(
        xs + (0.3 if max_q_s_a is not None else 0.0),
        n_s_a.cpu(),
        bottom=0,
        color="b",
        width=0.3,
        label="n_s_a",
    )
    ax.xaxis.set_major_locator(mticker.FixedLocator(np.arange(len(q_s_a))))
    ax.set_xticklabels(action_meanings, rotation=90)
    plt.subplots_adjust(bottom=0.2)
    ax.legend(loc="lower left")
    ax_n.legend(loc="lower right")
    ax.set_title("q_s_a and n_s_a")


def gen_video(video_stats, file_path):
    import cv2

    # Generate video
    imgs = []
    hw = video_stats["real_imgs"][0].shape[1]
    reset = False

    for i in range(len(video_stats["real_imgs"])):
        img = np.zeros(shape=(hw, hw * 2, 3), dtype=np.uint8)
        real_img = np.copy(video_stats["real_imgs"][i])
        real_img = np.swapaxes(np.swapaxes(real_img, 0, 2), 0, 1)
        im_img = np.copy(video_stats["im_imgs"][i])
        im_img = np.swapaxes(np.swapaxes(im_img, 0, 2), 0, 1)
        if video_stats["status"][i] == 1:
            # reset; yellow tint
            im_img[:, :, 0] = 255 * 0.3 + im_img[:, :, 0] * 0.7
            im_img[:, :, 1] = 255 * 0.3 + im_img[:, :, 1] * 0.7
        elif video_stats["status"][i] == 3:
            # force reset; red tint
            im_img[:, :, 0] = 255 * 0.3 + im_img[:, :, 0] * 0.7
        elif video_stats["status"][i] == 0:
            # real reset; blue tint
            im_img[:, :, 2] = 255 * 0.3 + im_img[:, :, 2] * 0.7
        img[:, :hw, :] = real_img
        img[:, hw:, :] = im_img
        img = np.flip(img, 2)
        imgs.append(img)

    enlarge_fcator = 3
    width = hw * 2 * enlarge_fcator
    height = hw * enlarge_fcator
    fps = 5

    path = os.path.join(file_path, "video.avi")
    fourcc = cv2.VideoWriter_fourcc(*"FFV1")
    video = cv2.VideoWriter(path, fourcc, float(fps), (width, height))
    video.set(cv2.CAP_PROP_BITRATE, 10000)  # set the video bitrate to 10000 kb/s
    for img in imgs:
        height, width, _ = img.shape
        new_height, new_width = height * enlarge_fcator, width * enlarge_fcator
        resized_img = cv2.resize(
            img, (new_width, new_height), interpolation=cv2.INTER_NEAREST
        )
        video.write(resized_img)
    video.release()


def print_im_actions(im_dict, action_meanings, print_stat=False):
    lookup_dict = {k: v for k, v in enumerate(action_meanings)}
    print_strs = []
    n, s = 1, ""
    reset = False
    for im, reset in zip(im_dict["pri"][:-1], im_dict["reset"][:-1]):
        s += lookup_dict[im.item()] + ", "
        if reset:
            s += "Reset" if reset == 1 else "FReset"
            print_strs.append("%d: %s" % (n, s))
            s = ""
            n += 1
    if not reset:
        print_strs.append("%d: %s" % (n, s[:-2]))
    if print_stat:
        for s in print_strs:
            print(s)
    return print_strs


def save_concatenated_image(buf1, buf2, strs, outdir, height=2500, width=3000):
    # Render the first figure onto a PIL image
    buf1.seek(0)
    img1 = Image.open(buf1)
    margin1 = 0

    # Render the second figure onto a PIL image
    buf2.seek(0)
    img2 = Image.open(buf2)
    margin2 = 0.1

    # Resize the images to have the desired width
    w1, h1 = img1.size
    new_width = int(width * (1 - 2 * margin1))
    new_height = int((new_width / w1) * h1)
    img1 = img1.resize((new_width, new_height))

    w2, h2 = img2.size
    new_width = int(width * (1 - 2 * margin2))
    new_height = int((new_width / w2) * h2)
    img2 = img2.resize((new_width, new_height))

    # Create a new image with the desired size
    result = Image.new("RGB", (width, height), color="white")

    # Paste the first image on the top
    result.paste(im=img1, box=(int(width * margin1), 0))

    # Paste the second image below the first one
    result.paste(im=img2, box=(int(width * margin2), img1.height))

    # Add the long string below the second image
    draw = ImageDraw.Draw(result)
    font_path = os.path.join(cv2.__path__[0], "qt", "fonts", "DejaVuSans.ttf")
    # font_path = os.path.join("C:\\", "Windows", "Fonts", "calibri.ttf")
    font = ImageFont.truetype(font_path, 34)

    y = (
        img1.height + img2.height + 10
    )  # Leave some space between the second image and the text
    font_box = font.getbbox("A")  # Get the height of a typical line of text
    line_height = font_box[3] - font_box[1]

    margint = 0.1
    for line in strs:
        # Wrap the line to fit the width of the image
        wrapper = textwrap.TextWrapper(width=int(width * (1 - 2 * margint)))
        lines = wrapper.wrap(line)

        for wrapped_line in lines:
            draw.text((width * margint, y), wrapped_line, font=font, fill="black")
            y += line_height

        # Add extra line spacing between paragraphs
        y += line_height
    # Save the concatenated image
    result.save(outdir)


def visualize(
    savedir,
    xpid,
    outdir,
    model_path="",
    plot=False,
    saveimg=True,
    savevideo=True,
    seed=-1,
    max_frames=-1,
):        
    savedir = savedir.replace("__project__", __project__)
    ckpdir = os.path.join(savedir, xpid)      
    if os.path.islink(ckpdir): ckpdir = os.readlink(ckpdir)  
    ckpdir =  os.path.abspath(os.path.expanduser(ckpdir))
    outdir = os.path.abspath(os.path.expanduser(outdir))

    max_eps_n = 1
    config_path = os.path.join(ckpdir, 'config_c.yaml')
    flags = util.create_flags(config_path, save_flags=False)
    if seed < 0:
        seed = np.random.randint(10000)
    env = Env(
        name=flags.name,
        env_n=1,
        base_seed=seed,        
        gpu=False,
        train_model=False,
        parallel=False,
        savedir=savedir,        
        xpid=xpid,
        ckp=True,
        return_x=True)

    if "Sokoban" in flags.name:
        action_meanings = ["NOOP", "UP", "DOWN", "LEFT", "RIGHT"]
    else:
        action_meanings = gym.make(flags.name).get_action_meanings()
    num_actions = env.num_actions

    env.seed([seed])
    print("Sampled seed: %d" % seed)    

    obs_space = env.observation_space
    action_space = env.action_space  
    actor_param = {
                "obs_space":obs_space,
                "action_space":action_space,
                "flags":flags
            }

    actor_net = ActorNet(**actor_param)
    checkpoint = torch.load(
        os.path.join(ckpdir, "ckp_actor.tar"), torch.device("cpu")
    )
    actor_net.set_weights(checkpoint["actor_net_state_dict"])
    actor_state = actor_net.initial_state(batch_size=1)
    print("Actor Net Real Steps: %d Steps: %d" % (checkpoint["real_step"],
                                                  checkpoint["step"])
                                                  )
    # create output folder
    n = 0
    while True:
        name = "%s-%d-%d" % (flags.xpid, checkpoint["real_step"], n)
        outdir_ = os.path.join(outdir, name)
        if not os.path.exists(outdir_):
            os.makedirs(outdir_)
            print(f"Outputting to {outdir_}")
            break
        n += 1
    outdir = outdir_

    # initalize env
    state = env.reset()
    env_out = init_env_out(state, flags)
    
    # some initial setting
    plt.rcParams.update({"font.size": 15})

    root_state = env_out.real_states
    tree_reps = util.decode_tree_reps(
        env_out.tree_reps, num_actions, flags.model_enc_type
    )
    end_gym_env_outs, end_titles = [], []
    ini_max_q = tree_reps["root_max_v"][0].item()

    step = 0
    real_step = 0
    returns, model_logits = (
        [],
        [],
    )
    im_list = ["pri_logits", "reset_logits", "pri", "reset"]
    im_dict = {k: [] for k in im_list}

    video_stats = {"real_imgs": [], "im_imgs": [], "status": [], "tree_reps": []}
    video_stats["real_imgs"].append(env_out.real_states[0, 0, -3:].numpy())
    video_stats["im_imgs"].append(video_stats["real_imgs"][-1])
    video_stats["status"].append(0)  # 0 for real step, 1 for reset, 2 for normal
    video_stats["tree_reps"].append({k: v.cpu().numpy() for k, v in tree_reps.items()})

    while len(returns) < max_eps_n:
        step += 1
        actor_out, actor_state, logits = actor_net(env_out, actor_state, extra_info=True)
        action = actor_out.action

        if env_out.step_status == 0:
            agent_v = actor_out.baseline[0, 0, 0]

        # additional stat record
        im_dict["pri_logits"].append(logits[0][:,0])
        im_dict["reset_logits"].append(logits[1][:,0])
        im_dict["pri"].append(actor_out.pri[:,0])
        im_dict["reset"].append(actor_out.reset[:,0])
        
        tree_reps_ = util.decode_tree_reps(
            env_out.tree_reps, num_actions, flags.model_enc_type
        )
        model_logits.append(tree_reps_["cur_logits"])
        
        state, reward, done, info = env.step(action[0], action[1])
        env_out = create_env_out(action, state, reward, done, info, flags)

        tree_reps = util.decode_tree_reps(
            env_out.tree_reps, num_actions, flags.model_enc_type
        )
        if (
            len(im_dict["reset"]) > 0
            and tree_reps["reset"]
            and not actor_out.reset
        ):
            im_dict["reset"][-1] = im_dict["reset"][-1].clone()
            im_dict["reset"][-1][:] = 3  # force reset

        xs = (torch.clamp(env_out.xs, 0, 1) * 255).to(torch.uint8)[0, 0]

        if env_out.step_status != 0 and (
            tree_reps["reset"] == 1 or env_out.step_status == 2
        ):
            title = "pred v: %.2f" % (tree_reps["cur_v"].item())
            title += " pred g: %.2f" % (tree_reps["root_trail_q"].item())
            end_gym_env_outs.append(xs.numpy())
            end_titles.append(title)

        # record data for generating video
        if env_out.step_status == 0:
            # real action
            video_stats["real_imgs"].append(xs[-3:].numpy())
            video_stats["status"].append(0)
        else:
            # imagainary action
            video_stats["real_imgs"].append(video_stats["real_imgs"][-1])
            video_stats["status"].append(2)
        video_stats["im_imgs"].append(xs[-3:].numpy())
        video_stats["tree_reps"].append(
            {k: v.cpu().numpy() for k, v in tree_reps.items()}
        )

        if im_dict["reset"][-1] in [1, 3]:
            # reset / force reset
            video_stats["real_imgs"].append(video_stats["real_imgs"][-1])
            video_stats["im_imgs"].append(video_stats["real_imgs"][-1])
            video_stats["status"].append(im_dict["reset"][-1].item())
            video_stats["tree_reps"].append(
                {k: v.cpu().numpy() for k, v in tree_reps.items()}
            )

        # visualize when a real step is made
        if (saveimg or plot) and env_out.step_status == 0:
            fig, axs = plt.subplots(1, 5, figsize=(50, 10))
            for k in im_list:
                if im_dict[k][0] is not None:
                    im_dict[k] = torch.concat(im_dict[k], dim=0)
                else:
                    im_dict[k] = None
            plot_gym_env_out(root_state[0], axs[0], title="Real State")
            plot_base_policies(
                torch.concat(model_logits), action_meanings=action_meanings, ax=axs[1]
            )
            plot_im_policies(
                **im_dict,
                action_meanings=action_meanings,
                one_hot=False,
                reset_ind=1,
                ax=axs[2],
            )
            plot_qn_sa(
                q_s_a=tree_reps_["root_qs_mean"][0],
                n_s_a=tree_reps_["root_ns"][0],
                action_meanings=action_meanings,
                max_q_s_a=tree_reps_["root_qs_max"][0],
                ax=axs[3],
            )
            model_policy_logits = tree_reps_["root_logits"][0]
            agent_policy_logits = logits[0][0, 0]
            action = torch.nn.functional.one_hot(
                actor_out.pri[0, 0], env.num_actions
            )
            plot_policies(
                [model_policy_logits, agent_policy_logits, action],
                ["model policy", "agent policy", "action"],
                action_meanings=action_meanings,
                ax=axs[4],
            )
            # plt.tight_layout()
            if saveimg:
                buf1 = BytesIO()
                plt.savefig(buf1, format="png")
            if plot:
                plt.show()
            plt.close()

            plot_multi_gym_env_out(end_gym_env_outs[:10], end_titles)
            if saveimg:
                buf2 = BytesIO()
                plt.savefig(buf2, format="png")
            if plot:
                plt.show()
            plt.close()

            log_str = "Step:%d (%d); return %.4f(%.4f) done %s real_done %s" % (
                real_step,
                step,
                env_out.episode_return[0, 0, 0],
                env_out.episode_return[0, 0, 1] if flags.im_cost > 0.0 else 0,
                "True" if env_out.done[0, 0] else "False",
                "True" if env_out.real_done[0, 0] else "False",
            )
            print(log_str)

            stat = f"Real Step: {real_step} Root v: {tree_reps_['root_v'][0].item():.2f} Actor v: {agent_v:.2f}"
            stat += f" Root Max Q: {tree_reps_['root_max_v'][0].item():.2f} Init. Root Max Q: {ini_max_q:.2f}"
            stat += f" Root Mean Q: {info['baseline'][0].item():.2f}"

            if flags.im_cost > 0.0:
                title += " im_return: %.4f" % env_out.episode_return[..., 1]

            if saveimg:
                im_action_strs = print_im_actions(
                    im_dict, action_meanings, print_stat=plot
                )
                save_concatenated_image(
                    buf1,
                    buf2,
                    [stat] + im_action_strs,
                    os.path.join(outdir, f"{real_step}.png"),
                )
                buf1.close()
                buf2.close()

            root_state = env_out.real_states
            im_dict = {k: [] for k in im_list}
            model_logits, end_gym_env_outs, end_titles = [], [], []
            ini_max_q = tree_reps["root_max_v"][0].item()

            real_step += 1

        if torch.any(env_out.real_done):
            step = 0
            new_rets = env_out.episode_return[env_out.real_done][:, 0].numpy()
            returns.extend(new_rets)
            print(
                "Finish %d episode: avg. return: %.2f (+-%.2f) "
                % (
                    len(returns),
                    np.average(returns),
                    np.std(returns) / np.sqrt(len(returns)),
                )
            )

        if max_frames >= 0 and real_step > max_frames:
            break

    if savevideo:
        video_stats["tree_reps"] = {
            k: np.concatenate([v[k] for v in video_stats["tree_reps"]], axis=0)
            for k in video_stats["tree_reps"][0].keys()
        }
        gen_video(video_stats, outdir)
        np.save(os.path.join(outdir, "video_stat.npy"), video_stats)

    return video_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Thinker visualization")
    parser.add_argument("--outdir", default="../test", help="Output directory.")
    parser.add_argument("--savedir", default="../logs/__project__", 
                        help="Checkpoint directory.")
    parser.add_argument("--xpid", default="latest", help="id of the run.")    
    parser.add_argument("--seed", default="-1", type=int, help="Base seed.")
    parser.add_argument(
        "--max_frames",
        default="-1",
        type=int,
        help="Max number of real frames to record",
    )
    flags = parser.parse_args()

    visualize(
        savedir=flags.savedir,
        xpid=flags.xpid,
        outdir=flags.outdir,
        plot=False,
        saveimg=True,
        savevideo=True,
        seed=flags.seed,
        max_frames=flags.max_frames,
    )
