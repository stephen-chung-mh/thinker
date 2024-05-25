import wandb
import argparse
import os
import shutil
import re

from thinker.util import __project__

parser = argparse.ArgumentParser(description='Download artifacts from W&B run.')
parser.add_argument('--project', type=str, default=__project__, help='Name of the project.')
parser.add_argument('--xpid', type=str, required=True, help='ID of the run.')
parser.add_argument('--output_path', type=str, default='../logs/__project__', help='Output directory to store downloaded files.')
parser.add_argument('--skip_download', action="store_true")

args = parser.parse_args()
m = re.match(r'^v\d+', args.xpid)
output_path = args.output_path
#if m: output_path = os.path.join(output_path, m[0])
sub_output_path = os.path.join(output_path, args.xpid)
sub_output_path = sub_output_path.replace("__project__", args.project)

if not args.skip_download:
    # Download all files from the run
    run = wandb.Api().run(f"{args.project}/{args.xpid}")
    for file in run.files():
        if file.name[-3:] == "gif": continue
        file.download(root=sub_output_path, replace=True)
        print(f"Downloaded file {file.name} to {os.path.join(sub_output_path, file.name)}")

# in new version all files is in xpid folder
source_path = os.path.join(sub_output_path, args.xpid)
if os.path.exists(source_path):
    # List all files in the source directory
    files = [f for f in os.listdir(source_path) if os.path.isfile(os.path.join(source_path, f))]
    
    for file_name in files:
        # Construct full file paths
        source_file = os.path.join(source_path, file_name)
        destination_file = os.path.join(sub_output_path, file_name)
        
        # Move the file with replacement
        if os.path.exists(destination_file):
            os.remove(destination_file)  # Remove the file at the destination if it exists
        shutil.move(source_file, sub_output_path)

print(f"File downloaded to {sub_output_path}")