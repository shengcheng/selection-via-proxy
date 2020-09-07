import os
from glob import glob
from typing import Tuple, Optional
from functools import partial
from tqdm import tqdm

import numpy as np
import torch
from torch import cuda
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from svp.common.datasets import DatasetWithIndex

from svp.common import utils
from svp.common.train import create_loaders
from svp.cifar.datasets import create_dataset
from svp.cifar.train import create_model_and_optimizer
from svp.common.selection import select
from svp.common.semisupervised import (generate_models,
                               check_different_models,
                               symlink_target_to_proxy,
                               symlink_to_precomputed_proxy,
                               validate_splits)


def semisupervised(run_dir: str = './run',

           datasets_dir: str = './data', dataset: str = 'cifar10',
           augmentation: bool = True,
           validation: int = 0, shuffle: bool = True,

           arch: str = 'preact20', optimizer: str = 'sgd',
           epochs: Tuple[int, ...] = (1, 90, 45, 45),
           learning_rates: Tuple[float, ...] = (0.01, 0.1, 0.01, 0.001),
           momentum: float = 0.9, weight_decay: float = 5e-4,
           batch_size: int = 128, eval_batch_size: int = 128,

           proxy_arch: str = 'preact20', proxy_optimizer: str = 'sgd',
           proxy_epochs: Tuple[int, ...] = (1, 90, 45, 45),
           proxy_learning_rates: Tuple[float, ...] = (0.01, 0.1, 0.01, 0.001),
           proxy_momentum: float = 0.9, proxy_weight_decay: float = 5e-4,
           proxy_batch_size: int = 128, proxy_eval_batch_size: int = 128,

           initial_subset: int = 1_000,
           selection_size: int = 4_000,
           selection_method: str = 'least_confidence',
           precomputed_selection: Optional[str] = None,
           train_target: bool = True,
           eval_target_at: Optional[Tuple[int, ...]] = None,

           cuda: bool = True,
           device_ids: Tuple[int, ...] = tuple(range(cuda.device_count())),
           num_workers: int = 0, eval_num_workers: int = 0,

           seed: Optional[int] = None, checkpoint: str = 'best',
           track_test_acc: bool = True):
    """
    Perform active learning on CIFAR10 and CIFAR100.

    If the model architectures (`arch` vs `proxy_arch`) or the learning rate
    schedules don't match, "selection via proxy" (SVP) is performed and two
    separate models are trained. The proxy is used for selecting which
    examples to label, while the target is only used for evaluating the
    quality of the selection. By default, the target model (`arch`) is
    trained and evaluated after each selection round. To change this behavior
    set `eval_target_at` to evaluate at a specific labeling budget(s) or set
    `train_target` to False to skip evaluating the target model. You can
    evaluate a series of selections later using the `precomputed_selection`
    option.

    Parameters
    ----------
    run_dir : str, default './run'
        Path to log results and other artifacts.
    datasets_dir : str, default './data'
        Path to datasets.
    dataset : str, default 'cifar10'
        Dataset to use in experiment (i.e., CIFAR10 or CIFAR100)
    augmentation : bool, default True
        Add data augmentation (i.e., random crop and horizontal flip).
    validation : int, default 0
        Number of examples from training set to use for valdiation.
    shuffle : bool, default True
        Shuffle training data before splitting into training and validation.
    arch : str, default 'preact20'
        Model architecture for the target model. `preact20` is short for
        ResNet20 w/ Pre-Activation.
    optimizer : str, default = 'sgd'
        Optimizer for training the target model.
    epochs : Tuple[int, ...], default (1, 90, 45, 45)
        Epochs for training the target model. Each number corresponds to a
        learning rate below.
    learning_rates : Tuple[float, ...], default (0.01, 0.1, 0.01, 0.001)
        Learning rates for training the target model. Each learning rate is
        used for the corresponding number of epochs above.
    momentum : float, default 0.9
        Momentum for SGD with the target model.
    weight_decay : float, default 5e-4
        Weight decay for SGD with the target model.
    batch_size : int, default 128
        Minibatch size for training the target model.
    eval_batch_size : int, default 128
        Minibatch size for evaluation (validation and testing) of the target
        model.
    proxy_arch : str, default 'preact20'
        Model architecture for the proxy model. `preact20` is short for
        ResNet20 w/ Pre-Activation.
    proxy_optimizer : str, default = 'sgd'
        Optimizer for training the proxy model.
    proxy_epochs : Tuple[int, ...], default (1, 90, 45, 45)
        Epochs for training the proxy model. Each number corresponds to a
        learning rate below.
    proxy_learning_rates : Tuple[float, ...], default (0.01, 0.1, 0.01, 0.001)
        Learning rates for training the proxy model. Each learning rate is
        used for the corresponding number of epochs above.
    proxy_momentum : float, default 0.9
        Momentum for SGD with the proxy model.
    proxy_weight_decay : float, default 5e-4
        Weight decay for SGD with the proxy model.
    proxy_batch_size : int, default 128
        Minibatch size for training the proxy model.
    proxy_eval_batch_size : int, default 128
        Minibatch size for evaluation (validation and testing) of the model
        proxy.
    initial_subset : int, default 1,000
        Number of randomly selected training examples to use for the initial
        labeled set.
    rounds : Tuple[int, ...], default (4,000, 5,000, 5,000, 5,000, 5,000)
        Number of unlabeled exampels to select in a round of labeling.
    selection_method : str, default least_confidence
        Criteria for selecting unlabeled examples to label.
    precomputed_selection : str or None, default None
        Path to timestamped run_dir of precomputed indices.
    train_target : bool, default True
        If proxy and target are different, train the target after each round
        of selection or specific rounds as specified below.
    eval_target_at : Tuple[int, ...] or None, default None
        If proxy and target are different and `train_target`, limit the
        evaluation of the target model to specific labeled subset sizes.
    cuda : bool, default True
        Enable or disable use of available GPUs
    device_ids : Tuple[int, ...], default True
        GPU device ids to use.
    num_workers : int, default 0
        Number of data loading workers for training.
    eval_num_workers : int, default 0
        Number of data loading workers for evaluation.
    seed : Optional[int], default None
        Random seed for numpy, torch, and others. If None, a random int is
        chosen and logged in the experiments config file.
    checkpoint : str, default 'best'
        Specify when to create a checkpoint for the model: only checkpoint the
        best performing model on the validation data or the training data if
        `validation == 0` ("best"), after every epoch ("all"), or only the last
        epoch of each segment of the learning rate schedule ("last").
    track_test_acc : bool, default True
        Calculate performance of the models on the test data in addition or
        instead of the validation dataset.'
    """
    # Set seeds for reproducibility.
    seed = utils.set_random_seed(seed)
    # Capture all of the arguments to save alongside the results.
    config = utils.capture_config(**locals())
    # Create a unique timestamped directory for this experiment.
    run_dir = utils.create_run_dir(run_dir, timestamp=config['timestamp'])
    utils.save_config(config, run_dir)
    # Update the computing arguments based on the runtime system.
    use_cuda, device, device_ids, num_workers = utils.config_run_env(
            cuda=cuda, device_ids=device_ids, num_workers=num_workers)

    # Create the training dataset.
    train_dataset = create_dataset(dataset, datasets_dir, train=True,
                                   augmentation=augmentation)
    # Verify there is enough training data for validation,
    #   the initial subset, and the selection rounds.
    validate_splits(train_dataset, validation, initial_subset, selection_size)

    # Create the test dataset.
    test_dataset = None
    if track_test_acc:
        test_dataset = create_dataset(dataset, datasets_dir, train=False,
                                      augmentation=False)

    # Calculate the number of classes (e.g., 10 or 100) so the model has
    #   the right dimension for its output.
    num_classes = len(set(train_dataset.train_labels))  # type: ignore

    # Split the training dataset between training and validation.
    unlabeled_pool, dev_indices = utils.split_indices(
        train_dataset, validation, run_dir, shuffle=shuffle)

    # Create the proxy to select which data points to label. If the
    #   selections were precomputed in another run or elsewhere, we can
    #   ignore this step.
    if precomputed_selection is None:
        # Use a partial so the appropriate model can be created without
        #   arguments.
        proxy_partial = partial(create_model_and_optimizer,
                                arch=proxy_arch,
                                num_classes=num_classes,
                                optimizer=proxy_optimizer,
                                learning_rate=proxy_learning_rates[0],
                                momentum=proxy_momentum,
                                weight_decay=proxy_weight_decay)

        # Create a directory for the proxy results to avoid confusion.
        proxy_run_dir = os.path.join(run_dir, 'proxy')
        os.makedirs(proxy_run_dir, exist_ok=True)
        # Create data loaders for validation and testing. The training
        #   data loader changes as labeled data is added, so it is
        #   instead a part of the proxy model generator below.
        _, proxy_dev_loader, proxy_test_loader = create_loaders(
            train_dataset,
            batch_size=proxy_batch_size,
            eval_batch_size=proxy_eval_batch_size,
            test_dataset=test_dataset,
            use_cuda=use_cuda,
            num_workers=num_workers,
            eval_num_workers=eval_num_workers,
            indices=(unlabeled_pool, dev_indices))

        # Create the proxy model generator (i.e., send data and get a
        #   trained model).
        proxy_generator = generate_models(
            proxy_partial, proxy_epochs, proxy_learning_rates,
            train_dataset,  proxy_batch_size,
            device, use_cuda,
            num_workers=num_workers,
            device_ids=device_ids,
            dev_loader=proxy_dev_loader,
            test_loader=proxy_test_loader,
            run_dir=proxy_run_dir,
            checkpoint=checkpoint)
        # Start the generator
        next(proxy_generator)

    # Check that the proxy and target are different models
    are_different_models = check_different_models(config)
    # Maybe create the target.
    if train_target:
        # If the proxy and target models aren't different, we don't
        #   need to create a separate model generator*.
        # * Unless the proxy wasn't created because the selections were
        #   precomputed (see above).
        if are_different_models or precomputed_selection is not None:
            # Use a partial so the appropriate model can be created
            #   without arguments.
            target_partial = partial(create_model_and_optimizer,
                                     arch=arch,
                                     num_classes=num_classes,
                                     optimizer=optimizer,
                                     learning_rate=learning_rates[0],
                                     momentum=momentum,
                                     weight_decay=weight_decay)

            # Create a directory for the target to avoid confusion.
            target_run_dir = os.path.join(run_dir, 'target')
            os.makedirs(target_run_dir, exist_ok=True)
            # Create data loaders for validation and testing. The training
            #   data loader changes as labeled data is added, so it is
            #   instead a part of the target model generator below.
            _, target_dev_loader, target_test_loader = create_loaders(
                train_dataset,
                batch_size=batch_size,
                eval_batch_size=eval_batch_size,
                test_dataset=test_dataset,
                use_cuda=use_cuda,
                num_workers=num_workers,
                eval_num_workers=eval_num_workers,
                indices=(unlabeled_pool, dev_indices))

            # Create the target model generator (i.e., send data and
            #   get a trained model).
            target_generator = generate_models(
                target_partial, epochs, learning_rates,
                train_dataset,  batch_size,
                device, use_cuda,
                num_workers=num_workers,
                device_ids=device_ids,
                dev_loader=target_dev_loader,
                test_loader=target_test_loader,
                run_dir=target_run_dir,
                checkpoint=checkpoint)
            # Start the generator
            next(target_generator)
        else:
            # Proxy and target are the same, so we can just symlink
            symlink_target_to_proxy(run_dir)

    # Perform active learning.
    if precomputed_selection is not None:
        assert train_target, "Must train target if selection is precomuted"
        assert os.path.exists(precomputed_selection)

        # Collect the files with the previously selected data.
        files = glob(os.path.join(precomputed_selection, 'proxy',
                     '*', 'labeled_*.index'))
        indices = [np.loadtxt(file, dtype=np.int64) for file in files]
        # Sort selections by length to replicate the order data was
        #   labeled.
        selections = sorted(
            zip(files, indices),
            key=lambda selection: len(selection[1]))  # type: ignore

        # Symlink proxy directories and files for convenience.
        symlink_to_precomputed_proxy(precomputed_selection, run_dir)

        # Train the target model on each selection.
        for file, labeled in selections:
            print('Load labeled indices from {}'.format(file))
            # Check whether the target model should be trained. If you
            #   have a specific labeling budget, you may not want to
            #   evaluate the target after each selection round to save
            #   time.
            should_eval = (eval_target_at is None or
                           len(eval_target_at) == 0 or
                           len(labeled) in eval_target_at)
            if should_eval:
                # Train the target model on the selected data.
                _, stats = target_generator.send(labeled)
                utils.save_result(stats, os.path.join(run_dir, "target.csv"))
    else:  # Select which points to label using the proxy.
        # Create initial random subset to train the proxy (warm start).
        labeled = np.random.permutation(unlabeled_pool)[:initial_subset]
        utils.save_index(labeled, run_dir,
                         'initial_subset_{}.index'.format(len(labeled)))

        # Train the proxy on the initial random subset
        model, stats = proxy_generator.send([labeled, train_dataset])
        utils.save_result(stats, os.path.join(run_dir, "proxy.csv"))


        # Select additional data to label from the unlabeled pool
        labeled, stats = select(model, train_dataset,
                                current=labeled,
                                pool=unlabeled_pool,
                                budget=selection_size,
                                method=selection_method,
                                batch_size=proxy_eval_batch_size,
                                device=device,
                                device_ids=device_ids,
                                num_workers=num_workers,
                                use_cuda=use_cuda)
        utils.save_result(stats, os.path.join(run_dir, 'selection.csv'))

        model, _ = proxy_generator.create_graph(run_dir)
        model = model.to(device)
        model = nn.DataParallel(model, device_ids=device_ids)

        # with open(os.path.join(run_dir, 'proxy.csv'), mode='a') as out:
        #     lines = out.readlines()
        # for line in lines:
                

        # Give the samples with fake labels
        fake_labeled = labeled[initial_subset:]
        train_sampler = SubsetRandomSampler(fake_labeled)
        train_loader = torch.utils.data.DataLoader(
            DatasetWithIndex(train_dataset), sampler=train_sampler,
            batch_size=batch_size, num_workers=num_workers, pin_memory=use_cuda, shuffle=False)
        wrapper_train_loader = tqdm(train_loader)

        model = model.to(device)
        train_labels = train_dataset.train_labels
        train_labels = np.array(train_labels)
        true_label = train_labels
        with torch.no_grad():
            for batch_size, (indices, inputs, targets) in enumerate(wrapper_train_loader):
                inputs = inputs.to(device)
                output = model(inputs)
                output = F.softmax(output, dim=1)
                _, output = torch.max(output, dim=1)
                ind = indices.data.cpu().numpy()
                output = output.data.cpu().numpy()
                train_labels[ind] = output

        train_labels = list(train_labels.astype(np.int))
        train_dataset.train_labels = train_labels
        stats_pseudo = {'pseudo_label_acc': np.sum(train_labels != true_label) / selection_size,
                        'selection size': selection_size,
                        }
        utils.save_result(stats_pseudo, os.path.join(run_dir, 'pseudo.csv'))


        # Train the proxy on the newly added data.
        model, stats = proxy_generator.send([labeled, train_dataset])
        utils.save_result(stats, os.path.join(run_dir, 'proxy.csv'))

        # Check whether the target model should be trained. If you
        #   have a specific labeling budget, you may not want to
        #   evaluate the target after each selection round to save
        #   time.
        should_eval = (eval_target_at is None or
                       len(eval_target_at) == 0 or
                       len(labeled) in eval_target_at)
        if train_target and should_eval and are_different_models:
            # Train the target model on the selected data.
            _, stats = target_generator.send(labeled)
            utils.save_result(stats, os.path.join(run_dir, "target.csv"))
