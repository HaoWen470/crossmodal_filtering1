#!/bin/bash
set -x

master_key=/juno/group/fusion_filtering/aws_keys/master.pem
user_key=/juno/group/fusion_filtering/aws_keys/_$USER.pem

if [[ ! -f $user_key ]]; then
    cp $master_key $user_key
    chmod 400 $user_key
fi


for ip in 3.21.70.36 3.12.140.140; do
    rsync --timeout=2 -Pavr -e "ssh -i $user_key" "ubuntu@$ip:~/multimodal_dpf2/{checkpoints,logs,metadata}" .
done
