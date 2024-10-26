#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, os, sys
# Add the root folder of the project to the path
ROOT_FOLDER = os.path.abspath(os.path.dirname(__file__) + '/../')
sys.path.append(ROOT_FOLDER)

import numpy as np
from Core.CDataSamplerInpainting import CDataSamplerInpainting
from Core.CDatasetLoader import CDatasetLoader
from Core.CTestInpaintingLoader import CTestInpaintingLoader
import Core.Utils as Utils
from collections import defaultdict
import time
from Core.CInpaintingTrainer import CInpaintingTrainer
import tqdm
import json
import wandb

def _eval(dataset, model):
    T = time.time()
    # Evaluate the model on the validation dataset
    loss = []
    for batchId in range(len(dataset)):
        batch = dataset[batchId]
        loss_value = model.eval(batch)
        loss.append(loss_value)
    loss = np.mean(loss)
    T = time.time() - T
    return loss, T

def evaluator(datasets, model, folder, args):
    losses = [np.inf] * len(datasets)  # Initialize with infinity
    def evaluate(onlyImproved=False, step=None):
        totalLoss = []
        eval_metrics = {}
        for i, dataset in enumerate(datasets):
            loss, T = _eval(dataset, model)
            dataset_id = ', '.join([str(x) for x in dataset.parametersIDs()])
            isImproved = loss < losses[i]
            if (not onlyImproved) or isImproved:
                print('Test %d / %d (%s) | %.2f sec | Loss: %.5f (%.5f).' % (
                    i + 1, len(datasets), dataset_id, T, loss, losses[i],
                ))
            if isImproved:
                print('Test %d / %d | Improved %.5f => %.5f,' % (
                    i + 1, len(datasets), losses[i], loss,
                ))
                modelFolder = os.path.join(folder, f"model-{dataset_id}")
                os.makedirs(modelFolder, exist_ok=True)
                # keep only the best model across all runs
                # name format: {model id}-{loss:.5f}-*.*
                all_files = os.listdir(modelFolder)
                all_losses = [f.split('-')[1] for f in all_files]
                all_losses = list(set(all_losses))
                print(f"Found losses: {all_losses}")
                for loss_file in all_losses:
                    if float(loss_file) > loss:
                        # remove all files with this loss in folder
                        to_remove = [os.path.join(modelFolder, f) for f in all_files if loss_file in f]
                        for f in to_remove:
                            os.remove(f)

                model.save(modelFolder, postfix='%.5f' % loss)
                losses[i] = loss
            totalLoss.append(loss)
            eval_metrics['eval_loss_(%s)' % dataset_id] = loss
        mean_loss = np.mean(totalLoss)
        if not onlyImproved:
            print('Mean loss: %.5f' % mean_loss)
        # Log evaluation metrics to wandb
        if step is not None:
            wandb.log(eval_metrics, step=step)
        return mean_loss
    return evaluate

def _modelTrainingLoop(model, dataset):
    def F(desc):
        history = defaultdict(list)
        # Use the tqdm progress bar
        with tqdm.tqdm(total=len(dataset), desc=desc) as pbar:
            dataset.on_epoch_start()
            for step in range(len(dataset)):
                sampled = dataset.sample()
                stats = model.fit(sampled)
                history['time'].append(stats['time'])
                for k in stats['losses'].keys():
                    history[k].append(stats['losses'][k])
                # Add stats to the progress bar (mean of each history)
                pbar.set_postfix({k: '%.5f' % np.mean(v) for k, v in history.items()})
                pbar.update(1)
            dataset.on_epoch_end()
        return {k: np.mean(v) for k, v in history.items()}
    return F

def _trainer_from(args):
    if args.trainer == 'default': return CInpaintingTrainer
    raise Exception('Unknown trainer: %s' % (args.trainer, ))

def main(args):
    wandb.init(project=args.wandb_project, config=vars(args))  # Initialize wandb
    timesteps = args.steps
    folder = os.path.join(args.folder, 'Data')

    stats = None
    with open(os.path.join(folder, 'remote', 'stats.json'), 'r') as f:
        stats = json.load(f)

    trainer = _trainer_from(args)
    trainDataset = CDatasetLoader(
        os.path.join(folder, 'remote'),
        stats=stats,
        sampling=args.sampling,
        samplerArgs=dict(
            batch_size=args.batch_size,
            minFrames=timesteps,
            maxT=1.0,
            defaults=dict(
                timesteps=timesteps,
                stepsSampling={'max frames': 10},
                # No augmentations by default
                pointsNoise=0.01, pointsDropout=0.01,
                eyesDropout=0.1, eyesAdditiveNoise=0.01, brightnessFactor=1.5, lightBlobFactor=1.5,
                targets=dict(keypoints=3, total=10),
            ),
            keys=['clean'],
        ),
        sampler_class=CDataSamplerInpainting,
        test_folders=['train.npz'],
    )
    model = dict(timesteps=timesteps, stats=stats)
    if args.model is not None:
        model['weights'] = dict(folder=folder, postfix=args.model, embeddings=args.embeddings)
    if args.modelId is not None:
        model['model'] = args.modelId

    model = trainer(**model)

    evalDatasets = [
        CTestInpaintingLoader(os.path.join(folderName, 'test-inpainting'))
        for folderName, _ in Utils.dataset_from_stats(stats, os.path.join(folder, 'remote'))
        if os.path.exists(os.path.join(folderName, 'test-inpainting'))
    ]
    eval_fn = evaluator(evalDatasets, model, folder, args)
    bestLoss = eval_fn()  # Evaluate loaded model
    bestEpoch = 0

    def evalWrapper(eval_fn):
        def f(epoch, onlyImproved=False, step=None):
            nonlocal bestLoss, bestEpoch
            newLoss = eval_fn(onlyImproved=onlyImproved, step=step)
            if newLoss < bestLoss:
                print('Improved %.5f => %.5f' % (bestLoss, newLoss))
                bestLoss = newLoss
                bestEpoch = epoch
                model.save(folder, postfix='%.5f' % newLoss)
            return
        return f

    eval_fn = evalWrapper(eval_fn)
    trainStep = _modelTrainingLoop(model, trainDataset)
    for epoch in range(args.epochs):
        metrics = trainStep(
            desc='Epoch %.*d / %d' % (len(str(args.epochs)), epoch, args.epochs),
        )
        wandb.log(metrics, step=epoch + 1)
        model.save(folder, postfix='latest')
        eval_fn(epoch, step=epoch + 1)
        print('Passed %d epochs since the last improvement (best: %.5f)' % (epoch - bestEpoch, bestLoss))
        if args.patience <= (epoch - bestEpoch):
            print('Early stopping')
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--model', type=str)
    parser.add_argument('--embeddings', default=False, action='store_true')
    parser.add_argument('--folder', type=str, default=ROOT_FOLDER)
    parser.add_argument('--modelId', type=str)
    parser.add_argument(
        '--trainer', type=str, default='default',
        choices=['default']
    )
    parser.add_argument(
        '--sampling', type=str, default='uniform',
        choices=['uniform', 'as_is'],
    )
    parser.add_argument('--wandb-project', type=str, default='alternative-input-reconstruction')

    main(parser.parse_args())
