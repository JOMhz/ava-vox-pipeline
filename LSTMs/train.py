import time, os, torch, argparse, warnings, glob

from dataLoader import train_loader, val_loader, train_loader_res_18, val_loader_res_18
from utils.tools import *
from ASD import ASD
from model.Model import ASD_Model
import re

def main():
    # This code is modified based on this [repository](https://github.com/TaoRuijie/TalkNet-ASD).
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description = "Model Training")
    # Training details
    parser.add_argument('--lr',           type=float, default=0.001, help='Learning rate')
    parser.add_argument('--lrDecay',      type=float, default=0.95,  help='Learning rate decay rate')
    parser.add_argument('--maxEpoch',     type=int,   default=25,    help='Maximum number of epochs')
    parser.add_argument('--testInterval', type=int,   default=1,     help='Test and save every [testInterval] epochs')
    parser.add_argument('--batchSize',    type=int,   default=800,  help='Dynamic batch size, default is 2000 frames')
    parser.add_argument('--nDataLoaderThread', type=int, default=12,  help='Number of loader threads')
    # Data path
    parser.add_argument('--dataPathAVA',  type=str, default="AVADataPath", help='Save path of AVA dataset')
    parser.add_argument('--savePath',     type=str, default="exps/exp1")
    # Data selection
    parser.add_argument('--evalDataType', type=str, default="val", help='Only for AVA, to choose the dataset for evaluation, val or test')
    # For download dataset only, for evaluation only
    parser.add_argument('--downloadAVA',     dest='downloadAVA', action='store_true', help='Only download AVA dataset and do related preprocess')
    parser.add_argument('--evaluation',      dest='evaluation', action='store_true', help='Only do evaluation by using pretrained model [pretrain_AVA_CVPR.model]')
    parser.add_argument('--ensemble',      dest='ensemble', action='store_true', help='Ensemble Test')
    
    args = parser.parse_args()
    # Data loader
    args = init_args(args)

    if args.downloadAVA == True:
        preprocess_AVA(args)
        quit()
    # loader = train_loader(trialFileName = args.trainTrialAVA, \
    #                       featurePath      = os.path.join(args.audioPathAVA , 'train'), \
    #                       **vars(args))
    loader = train_loader_res_18(trialFileName = args.trainTrialAVA, \
                          featurePath      = os.path.join(args.featurePathAVA , 'train'), \
                          **vars(args))
    trainLoader = torch.utils.data.DataLoader(loader, batch_size = 1, shuffle = True, num_workers = args.nDataLoaderThread, pin_memory = True)

    # loader = val_loader(trialFileName = args.evalTrialAVA, \
    #                     audioPath     = os.path.join(args.audioPathAVA , args.evalDataType), \
    #                     visualPath    = os.path.join(args.visualPathAVA, args.evalDataType), \
    #                     **vars(args))
    loader = val_loader_res_18(trialFileName = args.evalTrialAVA, \
                        featurePath     = os.path.join(args.featurePathAVA , 'val'), \
                        **vars(args))
    valLoader = torch.utils.data.DataLoader(loader, batch_size = 1, shuffle = False, num_workers = args.nDataLoaderThread, pin_memory = True)

    if args.ensemble == True:
        model_dir = "exps/ensemble/"
        model_files = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith('.model')]
        loaded_models = []
        for model_file in model_files:
            model_manager = ASD(**vars(args))
            model_manager.loadParameters(model_file)
            loaded_models.append(model_manager)

        for file in ['val', 'train']:
            csvSavePath = 'exps/ensemble/' + file + '_res.csv'
            origFileName = file + '_orig.csv'
            loaderFileName = file + '_loader.csv'
            loaderFileName = os.path.join(args.trialPathAVA, loaderFileName)
            origFileName = os.path.join(args.trialPathAVA, origFileName)
            loader = val_loader_res_18(trialFileName = loaderFileName, \
                        featurePath     = os.path.join(args.featurePathAVA , file), \
                        **vars(args))
            valLoader = torch.utils.data.DataLoader(loader, batch_size = 1, shuffle = False, num_workers = args.nDataLoaderThread, pin_memory = True)
            mAP = gnn_train_ensemble(loader = valLoader, model_managers = loaded_models, csvSavePath = csvSavePath, csvOrig = origFileName, **vars(args))
            print("mAP %2.2f%%"%(mAP))
        quit()

    if args.evaluation == True:
        s = ASD(**vars(args))
        s.loadParameters('weight/pretrain_AVA_CVPR.model')
        print("Model %s loaded from previous state!"%('pretrain_AVA_CVPR.model'))
        mAP = s.evaluate_network(loader = valLoader, **vars(args))
        print("mAP %2.2f%%"%(mAP))
        quit()

    modelfiles = glob.glob('%s/model_0*.model'%args.modelSavePath)
    modelfiles.sort()  
    if len(modelfiles) >= 1:
        print("Model %s loaded from previous state!"%modelfiles[-1])
        epoch = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][6:]) + 1
        s = ASD(epoch = epoch, **vars(args))
        s.loadParameters(modelfiles[-1])
    else:
        epoch = 1
        s = ASD(epoch = epoch, **vars(args))

    mAPs = []
    scoreFile = open(args.scoreSavePath, "a+")

    epochs = sorted([1, 5, 20, 35])
    rs = [1.5, 1.1, 0.9, 0.7]

    if epochs[-1] != args.maxEpoch:
        epochs.append(args.maxEpoch)
        rs.append(1)
    
    if epochs[0] != 1:
        epochs.append(1)
        rs.append(1)
    
    epochs_array = numpy.array(epochs)
    rs_array = numpy.array(rs)

    new_epoch = numpy.arange(1, 101)  # epochs from 1 to 100

    interpolated_r = numpy.interp(new_epoch, epochs, rs)
    mAPBestLoss = None
    max_r = 1.4
    min_r = 0.5
    r = interpolated_r[0]
    r_change_scale = 10
    totalLossDiff = 0
    while(1):
        loss, lr = s.train_network(epoch = epoch, loader = trainLoader, r = r, **vars(args))
        if mAPBestLoss:
            lossDiff = mAPBestLoss - loss
        else:
            lossDiff = 0
        if epoch % args.testInterval == 0:
            s.saveParameters(args.modelSavePath + "/model_%04d.model"%epoch)
            mAPs.append(s.evaluate_network(epoch = epoch, loader = valLoader, **vars(args)))
            print(time.strftime("%Y-%m-%d %H:%M:%S"), "%d epoch, mAP %2.2f%%, bestmAP %2.2f%%"%(epoch, mAPs[-1], max(mAPs)))
            scoreFile.write("%d epoch, LR %f, LOSS %f, mAP %2.2f%%, bestmAP %2.2f%%\n"%(epoch, lr, loss, mAPs[-1], max(mAPs)))
            scoreFile.flush()
            if mAPs[-1] >= max(mAPs) if mAPs else 0:  # Checks if current mAP is the best so far
                # r_change = -0.01  # Compute change, negative as mAP improved
                mAPBestLoss = loss
                totalLossDiff = 0
                r = interpolated_r[epoch]
                mAPBestR = r
            else:
                totalLossDiff += lossDiff  # Compute change, positive as mAP did not improve
                r = mAPBestR + totalLossDiff * r_change_scale
            r = max(min(r, max_r), min_r)
            
        if epoch >= args.maxEpoch:
            quit()

        epoch += 1

def gnn_train_ensemble(loader, model_managers, csvSavePath, csvOrig, **kwargs):
    # Set all models to evaluation mode
    for manager in model_managers:
        manager.eval()

        # This will store a list for each model's predictions and the ensemble predictions
    model_preds = [[] for _ in model_managers]
    ensemble_preds = []

    # Process each batch in the data loader
    for features, labels in tqdm.tqdm(loader):
        batch_scores = []
        # Process each model
        with torch.no_grad():
            for idx, manager in enumerate(model_managers):
                outsAV = manager.model.forward_audio_visual_backend(features[0].cuda())
                loss_labels = labels[0].reshape((-1)).cuda()
                _, predScore, _, _ = manager.lossAV.forward(outsAV, loss_labels)
                predScore = predScore[:, 1].detach().cpu().numpy()
                batch_scores.append(predScore)
                model_preds[idx].extend(predScore)

        # Compute ensemble score as the mean of predictions across all models for the current batch
        ensemble_score = numpy.mean(batch_scores, axis=0)
        ensemble_preds.extend(ensemble_score)


    # Prepare evaluation results
    evalLines = open(csvOrig).read().splitlines()[1:]
    labels = pandas.Series(['SPEAKING_AUDIBLE' for line in evalLines])
    scores_dict = {f'score_{i}': pandas.Series(preds) for i, preds in enumerate(model_preds)}
    scores_dict['score'] = pandas.Series(ensemble_preds)
    evalRes = pandas.read_csv(csvOrig)
    for key, series in scores_dict.items():
        evalRes[key] = series
    evalRes['label'] = labels
    evalRes.drop(['label_id', 'instance_id'], axis=1, inplace=True)
    evalRes.to_csv(csvSavePath, index=False)

    # Calculate mAP using an external script
    try:
        cmd = f"python -O utils/get_ava_active_speaker_performance.py -g {csvOrig} -p {csvSavePath}"
        proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        match = re.search(r'(\d+\.\d+)%', proc.stdout)
        if match:
            mAP = float(match.group(1))
        else:
            raise ValueError("mAP value not found in the output.", proc.stderr)

    except Exception as e:
        print(f"Error parsing mAP from output: {e}")
        mAP = 0.0  # Default to 0.0 if there's an error

    return mAP
def evaluate_ensemble(loader, model_managers, evalCsvSave, evalOrig, **kwargs):
    print(evalCsvSave)
    print(evalOrig)
    evalCsvSave = 'exps/ensemble/val_res.csv'
    trainCsvSave = 'exps/ensemble/train_res.csv'
    # train_orig = 
    # Set all models to evaluation mode
    for manager in model_managers:
        manager.eval()

    predScores = []

    # Process each batch in the data loader
    for features, labels in tqdm.tqdm(loader):
        batch_scores = []

        # Aggregate predictions from all models
        with torch.no_grad():
            for manager in model_managers:
                # print(features.shape)
                # print(labels.shape)
                outsAV = manager.model.forward_audio_visual_backend(features[0].cuda())
                loss_labels = labels[0].reshape((-1)).cuda()
                # print(outsAV.shape)
                _, predScore, _, _ = manager.lossAV.forward(outsAV, loss_labels)
                predScore = predScore[:, 1].detach().cpu().numpy()
                batch_scores.append(predScore)

        # Average or vote on predictions
        # Here we take the mean of predictions across all models
        ensemble_score = numpy.mean(batch_scores, axis=0)
        predScores.extend(ensemble_score)

    # Prepare evaluation results
    evalLines = open(evalOrig).read().splitlines()[1:]
    labels = pandas.Series(['SPEAKING_AUDIBLE' for line in evalLines])
    scores = pandas.Series(predScores)
    evalRes = pandas.read_csv(evalOrig)
    evalRes['score'] = scores
    evalRes['label'] = labels
    evalRes.drop(['label_id', 'instance_id'], axis=1, inplace=True)
    evalRes.to_csv(evalCsvSave, index=False)

    # Calculate mAP using an external script
    try:
        cmd = f"python -O utils/get_ava_active_speaker_performance.py -g {evalOrig} -p {evalCsvSave}"
        proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        match = re.search(r'(\d+\.\d+)%', proc.stdout)
        if match:
            mAP = float(match.group(1))
        else:
            raise ValueError("mAP value not found in the output.", proc.stderr)

    except Exception as e:
        print(f"Error parsing mAP from output: {e}")
        mAP = 0.0  # Default to 0.0 if there's an error

    return mAP


if __name__ == '__main__':
    main()
