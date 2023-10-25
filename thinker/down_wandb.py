import wandb
import argparse
import os
import re

from thinker.util import __project__

parser = argparse.ArgumentParser(description='Download artifacts from W&B run.')
parser.add_argument('--project', type=str, default=__project__, help='Name of the project.')
parser.add_argument('--xpid', type=str, required=True, help='ID of the run.')
parser.add_argument('--output_path', type=str, default='../logs/%s/'%__project__, help='Output directory to store downloaded files.')
parser.add_argument('--skip_download', action="store_true")

args = parser.parse_args()
m = re.match(r'^v\d+', args.xpid)
output_path = args.output_path
#if m: output_path = os.path.join(output_path, m[0])
output_path = os.path.join(output_path, args.xpid)

if not args.skip_download:
    # Download all files from the run
    run = wandb.Api().run(f"{args.project}/{args.xpid}")
    for file in run.files():
        if file.name[-3:] == "gif": continue
        file.download(root=output_path, replace=True)
        print(f"Downloaded file {file.name} to {os.path.join(output_path, file.name)}")

print(f"File downloaded to {output_path}")