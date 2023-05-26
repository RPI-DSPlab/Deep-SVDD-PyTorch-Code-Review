#!/bin/bash

# Main python script path
main_script="main.py"

# Common parameters
cifar="cifar10"
model="cifar10_LeNet"
data_dir="../data"
objective="one-class"
lr="0.0001"
n_epochs="150"
lr_milestone="50"
batch_size="200"
weight_decay="0.5e-6"
pretrain="True"
ae_lr="0.0001"
ae_n_epochs="350"
ae_lr_milestone="250"
ae_batch_size="200"
ae_weight_decay="0.5e-6"

# Iterating over 10 classes of CIFAR10
for class in {0..9}
do
    # Prepare the log directory path
    log_dir="../log/cifar10_test/class${class}"

    # Create log directory if it does not exist
    mkdir -p "${log_dir}"

    # Run the python command for each class
    python ${main_script} ${cifar} ${model} ${log_dir} ${data_dir} --objective ${objective} --lr ${lr} --n_epochs ${n_epochs} --lr_milestone ${lr_milestone} --batch_size ${batch_size} --weight_decay ${weight_decay} --pretrain ${pretrain} --ae_lr ${ae_lr} --ae_n_epochs ${ae_n_epochs} --ae_lr_milestone ${ae_lr_milestone} --ae_batch_size ${ae_batch_size} --ae_weight_decay ${ae_weight_decay} --normal_class ${class};
done

