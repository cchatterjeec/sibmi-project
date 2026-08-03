#!/bin/bash
#SBATCH -p priority     # Change to desired partition
#SBATCH --mem=64g        # Change to desired RAM
#SBATCH --time=06:00:00 # Change to desired wall-time
#SBATCH -c 4            # Change to desired CPU cores
#SBATCH -o vscode.out   # Output file name

set -o errexit -o nounset -o pipefail
MY_SCRATCH="/n/groups/patel/chandrima"
mkdir -p $MY_SCRATCH
echo $MY_SCRATCH

#If not done yet, obtain the tarball and untar it in $MY_SCRATCH location to obtain the
#executable, code
if [ -e "$MY_SCRATCH/code" ]; then
    echo "using $MY_SCRATCH/code"
else
    echo "install a new version of code"
    curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' | tar -C $MY_SCRATCH -xzf -
fi

#And run it using the provider of your choice
$MY_SCRATCH/code tunnel user login --provider microsoft
#$MY_SCRATCH/code tunnel user login --provider github

#Accept the license terms & launch the tunnel
$MY_SCRATCH/code tunnel --accept-server-license-terms --name o2tunnel
