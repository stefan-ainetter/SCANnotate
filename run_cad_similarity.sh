#!/bin/bash
DIR_PATH=$(cd $(dirname "${BASH_SOURCE:-$0}") && pwd)
file="render_pipeline/CAD_retrieval_similarity.py"
execute_path=$DIR_PATH/$file

for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

echo "split = $split"
echo "gpu = $gpu"
echo "config = $config"

python3 $execute_path --config=$config --data_split=$split --device=$gpu
