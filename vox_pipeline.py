import random
import pandas as pd
import time, os, torch, argparse, warnings, glob
from sklearn.model_selection import KFold
from training.audio_visual_data_loader import ValLoader, ValLoaderAudio, ValLoaderVideo
from training.fast_loader import FastTrainLoader, FastValLoader
from training.train_eval import VoxSense
from training.ASD import ASD
from utils import get_fold_video_ids, get_loader, init_args, parse_args

def fast_main(args_update={}):
    args = parse_args(args_update=args_update)
    args = init_args(args)

    # The structure of this code is learnt from [9]
    warnings.filterwarnings("ignore")


    

    epoch = 1
    # model = VoxSense(epoch = epoch, **vars(args))
    model = ASD(epoch = epoch, **vars(args))
    model = model.cuda()

    print(f"Starting from epoch: {epoch}")

    mAPs = []
    
    # Define the path for the CSV file
    csv_file_path = os.path.join(args.save_path, "training_metrics.csv")
    metrics_df = pd.DataFrame()

    args_dict = vars(args)
    args_df = pd.DataFrame([args_dict])
    args_csv_file_path = os.path.join(args.save_path, "args_configuration.csv")
    args_df.to_csv(args_csv_file_path, index=False)

    # Seed the random number generator for reproducibility
    random.seed(args.random_seed)

    train_video_ids = pd.read_csv(args.train_trial_list_AVA, header=None).squeeze().tolist()

    reduced_size = max(int(len(train_video_ids) * args.scale_factor), args.n_splits)
    reduced_train_video_ids = random.sample(train_video_ids, reduced_size)

    # Now, use adjusted_n_splits for K-Fold
    fold_video_ids = get_fold_video_ids(reduced_train_video_ids, n_splits=args.n_splits)

    for epoch in range(1, args.max_epoch + 1):
        start_time_epoch = time.time()
        # Determine the current fold based on the epoch
        current_fold = ((epoch - 1) // 3) % args.n_splits
        train_ids, val_ids = fold_video_ids[current_fold]
        # Initialize loaders with current fold's video IDs
        # train_loader = get_loader(args, eval=False, current_fold_video_ids=train_ids)
        # val_loader = get_loader(args, eval=True, current_fold_video_ids=val_ids)

        loader = FastTrainLoader(trial_file_name=args.train_trial_AVA,
                  audio_path=os.path.join(args.audio_path_AVA, 'train'),
                  visual_path=os.path.join(args.visual_path_AVA, 'train'),
                  current_fold_video_ids=train_ids,
                  **vars(args))
        train_loader = torch.utils.data.DataLoader(loader, batch_size = 1, shuffle = True, num_workers = args.n_data_loader_thread, pin_memory = True)

        loader = FastValLoader(trial_file_name=args.train_trial_AVA,
                  audio_path=os.path.join(args.audio_path_AVA, 'train'),
                  visual_path=os.path.join(args.visual_path_AVA, 'train'),
                  current_fold_video_ids=val_ids,
                  **vars(args))
        val_loader = torch.utils.data.DataLoader(loader, batch_size = 1, shuffle = False, num_workers = 64, pin_memory = True)

        # Convert loaders to DataLoader objects
        # train_loader = torch.utils.data.DataLoader(train_loader, batch_size=1, shuffle=True, num_workers=args.n_data_loader_thread, pin_memory=True)
        # val_loader = torch.utils.data.DataLoader(val_loader, batch_size=1, shuffle=False, num_workers=args.n_data_loader_thread, pin_memory=True)
        
        train_metrics = model.train_network(epoch=epoch, loader=train_loader, max_epochs=args.max_epoch, **vars(args))
        # train_metrics = {}
        if epoch % args.test_interval == 0:
            model.save_parameters(f"{args.model_save_path}/model_{epoch:04d}.model")

            eval_metrics = model.evaluate_network(epoch=epoch, loader=val_loader, **vars(args))
            
            mAPs.append(eval_metrics['mAP'])
            
            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            best_mAP = max(mAPs)
            epoch_duration = time.time() - start_time_epoch

            new_row = pd.DataFrame([{"epoch": epoch, **train_metrics, **eval_metrics, 'epoch_duration': epoch_duration}])
            metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)
            metrics_df.to_csv(csv_file_path, index=False)

            print(f"{current_time} - Epoch: {epoch}, mAP: {eval_metrics['mAP']:.2f}%, best mAP: {best_mAP:.2f}%")

        epoch += 1

    if args.use_audio and args.use_video:
            loader = ValLoader(trial_file_name = args.eval_trial_AVA, \
                        audio_path     = os.path.join(args.audio_path_AVA , args.eval_data_type), \
                        visual_path    = os.path.join(args.visual_path_AVA, args.eval_data_type), \
                        **vars(args))
    elif args.use_audio:
        loader = ValLoaderAudio(trial_file_name = args.eval_trial_AVA, \
                    audio_path     = os.path.join(args.audio_path_AVA , args.eval_data_type), \
                    visual_path    = os.path.join(args.visual_path_AVA, args.eval_data_type), \
                    **vars(args))
    elif args.use_video:
        loader = ValLoaderVideo(trial_file_name = args.eval_trial_AVA, \
                    audio_path     = os.path.join(args.audio_path_AVA , args.eval_data_type), \
                    visual_path    = os.path.join(args.visual_path_AVA, args.eval_data_type), \
                    **vars(args))
    
    val_loader = torch.utils.data.DataLoader(loader, batch_size = 1, shuffle=False, num_workers=args.n_data_loader_thread)

    # Evaluate the model on the evaluation dataset
    test_metrics = model.evaluate_network(loader=val_loader, epoch=epoch, full_test=True, **vars(args))
    # test_metrics = {}
    test_row = pd.DataFrame([{'epoch': 'test', **test_metrics}])
    metrics_df = pd.concat([metrics_df, test_row], ignore_index=True)
    
    print(f"Test mAP: {test_metrics['mAP']:.2f}%")

    metrics_df.to_csv(csv_file_path, index=False)


def main(args_update={}):
    args = parse_args(args_update=args_update)
    args = init_args(args)

    # The structure of this code is learnt from [9]
    warnings.filterwarnings("ignore")


    

    epoch = 1
    model = VoxSense(epoch = epoch, **vars(args))
    # model = ASD(epoch = epoch, **vars(args))
    model = model.cuda()

    print(f"Starting from epoch: {epoch}")

    mAPs = []
    
    # Define the path for the CSV file
    csv_file_path = os.path.join(args.save_path, "training_metrics.csv")
    metrics_df = pd.DataFrame()

    args_dict = vars(args)
    args_df = pd.DataFrame([args_dict])
    args_csv_file_path = os.path.join(args.save_path, "args_configuration.csv")
    args_df.to_csv(args_csv_file_path, index=False)

    # Seed the random number generator for reproducibility
    random.seed(args.random_seed)

    train_video_ids = pd.read_csv(args.train_trial_list_AVA, header=None).squeeze().tolist()

    reduced_size = max(int(len(train_video_ids) * args.scale_factor), args.n_splits)
    reduced_train_video_ids = random.sample(train_video_ids, reduced_size)

    # Now, use adjusted_n_splits for K-Fold
    fold_video_ids = get_fold_video_ids(reduced_train_video_ids, n_splits=args.n_splits)

    for epoch in range(1, args.max_epoch + 1):
        start_time_epoch = time.time()
        # Determine the current fold based on the epoch
        current_fold = ((epoch - 1) // 3) % args.n_splits
        train_ids, val_ids = fold_video_ids[current_fold]
        
        # Initialize loaders with current fold's video IDs
        train_loader = get_loader(args, eval=False, current_fold_video_ids=train_ids)
        val_loader = get_loader(args, eval=True, current_fold_video_ids=val_ids)

        # Convert loaders to DataLoader objects
        train_loader = torch.utils.data.DataLoader(train_loader, batch_size=1, shuffle=True, num_workers=args.n_data_loader_thread, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_loader, batch_size=1, shuffle=False, num_workers=args.n_data_loader_thread, pin_memory=True)
        
        train_metrics = model.train_network(epoch=epoch, loader=train_loader, max_epochs=args.max_epoch, **vars(args))
        # train_metrics = {}
        if epoch % args.test_interval == 0:
            model.save_parameters(f"{args.model_save_path}/model_{epoch:04d}.model")

            eval_metrics = model.evaluate_network(epoch=epoch, loader=val_loader, **vars(args))
            
            mAPs.append(eval_metrics['mAP'])
            
            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            best_mAP = max(mAPs)
            epoch_duration = time.time() - start_time_epoch

            new_row = pd.DataFrame([{"epoch": epoch, **train_metrics, **eval_metrics, 'epoch_duration': epoch_duration}])
            metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)
            metrics_df.to_csv(csv_file_path, index=False)

            print(f"{current_time} - Epoch: {epoch}, mAP: {eval_metrics['mAP']:.2f}%, best mAP: {best_mAP:.2f}%")

        epoch += 1

    if args.use_audio and args.use_video:
        loader = ValLoader(trial_file_name = args.eval_trial_AVA, \
                    audio_path     = os.path.join(args.audio_path_AVA , args.eval_data_type), \
                    visual_path    = os.path.join(args.visual_path_AVA, args.eval_data_type), \
                    **vars(args))
    elif args.use_audio:
        loader = ValLoaderAudio(trial_file_name = args.eval_trial_AVA, \
                    audio_path     = os.path.join(args.audio_path_AVA , args.eval_data_type), \
                    visual_path    = os.path.join(args.visual_path_AVA, args.eval_data_type), \
                    **vars(args))
    elif args.use_video:
        loader = ValLoaderVideo(trial_file_name = args.eval_trial_AVA, \
                    audio_path     = os.path.join(args.audio_path_AVA , args.eval_data_type), \
                    visual_path    = os.path.join(args.visual_path_AVA, args.eval_data_type), \
                    **vars(args))
    
    val_loader = torch.utils.data.DataLoader(loader, batch_size = 1, shuffle=False, num_workers=args.n_data_loader_thread)

    # Evaluate the model on the evaluation dataset
    test_metrics = model.evaluate_network(loader=val_loader, epoch=epoch, full_test=True, **vars(args))
    # test_metrics = {}
    test_row = pd.DataFrame([{'epoch': 'test', **test_metrics}])
    metrics_df = pd.concat([metrics_df, test_row], ignore_index=True)
    
    print(f"Test mAP: {test_metrics['mAP']:.2f}%")

    metrics_df.to_csv(csv_file_path, index=False)

if __name__ == "__main__":
    
    # main(args_update={"batch_size": 1000, "n_data_loader_thread": 12, "scale_factor": 0.001})
    fast_main(args_update={"batch_size": 1000, "n_data_loader_thread": 6, "scale_factor": 1})