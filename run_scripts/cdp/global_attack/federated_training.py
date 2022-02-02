import subprocess
import multiprocessing
import time


def execute_command_wait(cmd):
    subprocess.call(cmd)


def execute_command(cmd):
    subprocess.Popen(cmd)


ec2_instances = ["ec2-xy-xyz..."]  # list of EC2 Instances
key = "/~/.ssh/aws.pem"  # path to AWS Key
clients = 4  # number of clients training in parallel
target_lr = 0.001  # target model learning rate
target_b = 64  # target model batch size
attack_b = 32  # attack model batch size
attack_lr = 0.0007  # attack model learning rate
model = "texas"  # dataset
seeds = ["42"]  # list of seeds
inference = "wb"  # wb = membership inference, ai = attribute inference
output_sizes = ["100"]  # model configurations
optimize = 0  # if set to 1, gaussian optimizer is executed for a range of hyperparams
noise = ["0.25"]  # sigma CDP noise
norm_clip = 2  # Norm Clip


def bootstrap_instance(seed, i, output_size, noise, connection_string):
    noise_str = str(noise).replace(".", "_")
    experiment_name = f"{model}{output_size}_cdp_{noise_str}_s_{seed}"
    if model == "texas" or model == "purchases":
        dataset = f"./data/shokri_{model}_{output_size}_classes.npz"
    else:
        dataset = f"./data/{model}_{output_size}_classes.npz"

    output = f"./experiments/global/{experiment_name}"

    start_server_command = " ".join(
        ["nohup", "python", "-m",
         "libs.FederatedFramework.core.experiments.run_script", "--output",
         f"{output}/target", "--model", model,
         "--batch_size",
         str(target_b), "--clients", str(clients), "--min_connected", str(clients), "--parallel_workers",
         "4",
         "--epochs", "1", "--experiment", experiment_name, "--save_epochs", "5",
         "--isServer", "--learning_rate", str(target_lr),
         "--alternative_dataset", dataset,
         "--output_size", str(output_size), "--norm_clip", str(norm_clip), "--noise_multiplier",
         str(noise), "--seed", str(seed), "--port", str(5000 + i)])
    start_clients_command = " ".join(
        ["nohup", "python", "-m",
         "libs.FederatedFramework.core.experiments.run_script", "--output",
         f"{output}/target", "--model", model, "--batch_size",
         str(target_b), "--clients", str(clients), "--experiment", experiment_name, "--isClient", "--learning_rate",
         str(target_lr),
         "--alternative_dataset", dataset,
         "--output_size", str(output_size), "--norm_clip", str(norm_clip), "--noise_multiplier",
         str(noise), "--seed", str(seed), "--port", str(5000 + i)])
    start_mi_command = " ".join(
        ["nohup", "python", f"./FIA/run_scripts/nodp/global_attack/train_{inference}.py", "--path",
         output,
         "--dataset", dataset, "--output_size", output_size, "--seed", str(seed), "--batch_size", str(attack_b),
         "--learning_rate", str(attack_lr), "--optimize", optimize]
    )
    print(f"start rsync on {connection_string}")
    execute_command_wait(
        ["rsync", "-amv", "--stats", "--exclude", ".git", "--exclude", ".fuse*", "--exclude",
         "*.h5", "--exclude",
         "*.hdf5",
         "--exclude", "*.json", "-e", f"ssh -o StrictHostKeyChecking=no -i {key}",
         "../FIA",
         f"{connection_string}:/home/ubuntu/"])
    print(f"start training on {connection_string}")
    execute_command(
        ["ssh", "-o", "StrictHostKeyChecking=no", "-i", key, connection_string,
         f"cd /home/ubuntu/FIA;"
         f"mkdir -p {output}/target;"
         f"mkdir -p ./logs;"
         f"pip install -r requirements.txt;"
         f"{start_clients_command} >/dev/null 2>./logs/clients{i}.txt &"
         f"{start_server_command} >/dev/null 2>./logs/server{i}.txt;"
         f"sleep 280;"
         f"{start_mi_command} >./logs/mia_{output_size}_{noise_str}.txt;"
         ])


i = 0
for seed in seeds:
    for output_size in output_sizes:
        for nm in noise:
            p = multiprocessing.Process(target=bootstrap_instance, args=(seed, i, output_size, nm,
                                                                         f"ubuntu@{ec2_instances[i % len(ec2_instances)]}",))
            p.start()
            time.sleep(20)
            i += 1
