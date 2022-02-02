import signal
import sys

import argparse
import importlib
import os
import logging
from ..entities.experiment import Experiment

"""
    This script runs federated experiments
"""

assert os.getcwd().endswith("FIA"), "script should be started from home folder"

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Target model training")
    parser.add_argument("--seed", "-s", default=42, type=int, help="Random Seed")
    parser.add_argument("--batch_size", "-b", default=128, type=int, help="Mini Batch size (default 128)")
    parser.add_argument("--epochs", "-es", default=1, type=int, help="Local Epoch per Round")
    parser.add_argument("--experiment", "-e", type=str, required=True, help="Experiment name")
    parser.add_argument("--model", "-m", required=True, type=str, help="model used to train")
    parser.add_argument("--clients", "-c", type=int, default=4, help="number of clients")
    parser.add_argument("--isClient", "-isc", action='store_true', help="is client")
    parser.add_argument("--isServer", "-iss", action='store_true', help="is server")
    parser.add_argument("--hasAdversaryClient", "-a", action='store_true',
                        help="Does this experiment include an adversarial client")
    parser.add_argument("--min_connected", "-mc", default=4, type=int, help="minimum connected clients before training")
    parser.add_argument("--parallel_workers", "-pw", default=2, type=int,
                        help="parallel working clients. parallel_workers = C*clients. C from paper by McMahan")
    parser.add_argument('--save_epochs', type=int, help='every nth epoch should save a snapshot')
    parser.add_argument("--output", "-o", required=True, type=str, help="output path")
    parser.add_argument("--noise_multiplier", "-nm", type=float, default=0.0)
    parser.add_argument("--norm_clip", "-nc", type=float, default=1.0)
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.001)
    parser.add_argument("--output_size", "-os", type=int, default=100)
    parser.add_argument("--port", "-p", type=int, default=5000)
    parser.add_argument("--alternative_dataset", "-ad", type=str, default=None)
    parser.set_defaults(hasAdversaryClient=False)
    args = parser.parse_args()
    assert not (
            args.isServer and args.hasAdversaryClient), "hasAdversaryClient can only be used in conjunction with isClient"
    assert args.hasAdversaryClient is False or args.save_epochs is not None, "hasAdversaryClient can only be used in conjunction with save epochs"
    assert args.save_epochs is None or args.save_epochs > 0, "save epoch should be greater than zero"
    assert args.isClient or args.isServer, "server or client required"
    assert args.isClient or args.min_connected <= args.clients, \
        "minimum connections must be equal or smaller as clients"
    assert os.path.isdir(args.output), f"{args.output} output should be an existing directory"
    assert args.alternative_dataset is None or os.path.exists(
        args.alternative_dataset), f"alternative data {args.alternative_dataset} set (e.g. ldp perturbed) should exist"
    module_import = importlib.import_module(name=f'.{args.model}.{args.model}',
                                            package='libs.FederatedFramework.core.experiments')
    config = {"noise_multiplier": args.noise_multiplier, "norm_clip": args.norm_clip, "batch_size": args.batch_size,
              "learning_rate": args.learning_rate, "alternative_dataset": args.alternative_dataset,
              "output_size": args.output_size}

    module = getattr(module_import, f'{args.model.capitalize()}Experiment')(config)
    assert isinstance(module, Experiment), "import should be a proper experiment"
    if args.save_epochs is not None:
        args.save_epochs = int(args.save_epochs)
    if args.isServer:
        print(f"listening on 127.0.0.1:{args.port}")
        module.start_server(args.clients + 1, args.seed, args.output, args.save_epochs,
                            args.parallel_workers, args.min_connected, args.batch_size, args.epochs, args.port)

    if args.isClient:
        module.start_clients(args.clients + 1, args.seed, args.output, args.hasAdversaryClient, args.save_epochs, args.port)


    def signal_handler(sig, frame):
        print('Keyboard interrupt!')
        module.stop_all()
        sys.exit(0)


    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
