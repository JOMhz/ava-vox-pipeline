import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, time, os, subprocess, re
import numpy as np
import pandas as pd

from model.vox.vox_model import VoxSenseModel
from model.vox.loss import lossA, lossAV, lossV


class VoxSense(nn.Module):
    def __init__(self, lr=0.0001, lr_decay=0.95, **kwargs):
        super(VoxSense, self).__init__()
        self.config = kwargs
        self.model = VoxSenseModel(temporal=self.config['use_temporal'])
        self.lossAV = lossAV()
        self.lossA = lossA()
        self.lossV = lossV()
        self.optim = torch.optim.Adam(self.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=1, gamma=lr_decay)
        total_params = sum(param.numel() for param in self.model.parameters()) / 1024 / 1024
        print(f"{time.strftime('%m-%d %H:%M:%S')} Model parameter count = {total_params:.2f}M")

    def train_audio(self, loader, epoch, total_batches, print_update_interval):
        index, top1, loss = 0, 0, 0
        for num, (audio_feature, _, labels) in enumerate(loader, start=1):
            self.zero_grad()
            audio_embed = self.model.forward_audio_frontend(audio_feature[0].cuda())
            labels = labels[0].reshape((-1)).cuda()
            
            outsA = self.model.forward_audio_backend(audio_embed)
            nlossA, _, _, precision = self.lossA.forward(outsA, labels)
            nloss = nlossA
            loss += nloss.detach().cpu().numpy()
            top1 += precision
            nloss.backward()
            self.optim.step()
            index += len(labels)
            
            # Dynamic output
            progress = 100 * num / total_batches
            avg_loss = loss / num
            avg_acc = 100 * top1 / index
            if num % print_update_interval == 0 or num == total_batches - 1:
                print(f"\rEpoch: [{epoch}/{self.config['max_epoch']}], Lr: {self.lr:.5f}, Progress: {progress:.2f}%, Loss: {avg_loss:.5f}, ACC: {avg_acc:.2f}% Elapsed time: {time.time() - self.start_time:.2f} seconds.", end='')
        train_loss = loss / total_batches
        train_acc = 100 * top1 / index
        return train_loss, train_acc
    

    def train_video(self, loader, epoch, total_batches, print_update_interval):
        index, top1, loss = 0, 0, 0
        for num, (_, visual_feature, labels) in enumerate(loader, start=1):
            self.zero_grad()
            visual_embed = self.model.forward_visual_frontend(visual_feature[0].cuda())
            labels = labels[0].reshape((-1)).cuda()
            
            outsV = self.model.forward_visual_backend(visual_embed)
            nlossV, _, _, precision = self.lossV.forward(outsV, labels)
            nloss = nlossV
            loss += nloss.detach().cpu().numpy()
            top1 += precision
            nloss.backward()
            self.optim.step()
            index += len(labels)
            
            # Dynamic output
            progress = 100 * num / total_batches
            avg_loss = loss / num
            avg_acc = 100 * top1 / index
            if num % print_update_interval == 0 or num == total_batches - 1:
                print(f"\rEpoch: [{epoch}/{self.config['max_epoch']}], Lr: {self.lr:.5f}, Progress: {progress:.2f}%, Loss: {avg_loss:.5f}, ACC: {avg_acc:.2f}% Elapsed time: {time.time() - self.start_time:.2f} seconds.", end='')
        train_loss = loss / total_batches
        train_acc = 100 * top1 / index
        return train_loss, train_acc


    def train_audio_video(self, loader, epoch, total_batches, print_update_interval):
        index, top1, loss = 0, 0, 0
        for num, (audio_feature, visual_feature, labels) in enumerate(loader, start=1):
            self.zero_grad()
            audio_embed = self.model.forward_audio_frontend(audio_feature[0].cuda())
            visual_embed = self.model.forward_visual_frontend(visual_feature[0].cuda())
            labels = labels[0].reshape((-1)).cuda()
            audio_embed, visual_embed = self.model.forward_cross_attention(audio_embed, visual_embed)
            outsAV= self.model.forward_audio_visual_backend(audio_embed, visual_embed)  
            outsA = self.model.forward_audio_backend(audio_embed)
            outsV = self.model.forward_visual_backend(visual_embed)
            
            # Loss
            nlossAV, _, _, precision = self.lossAV.forward(outsAV, labels)
            nlossA, _, _, _ = self.lossA.forward(outsA, labels)
            nlossV, _, _, _ = self.lossV.forward(outsV, labels)
            nloss = nlossAV + 0.4 * nlossA + 0.4 * nlossV
            loss += nloss.detach().cpu().numpy()
            top1 += precision
            nloss.backward()
            self.optim.step()
            index += len(labels)
            
            # Dynamic output
            progress = 100 * num / total_batches
            avg_loss = loss / num
            avg_acc = 100 * top1 / index
            if num % print_update_interval == 0 or num == total_batches - 1:
                print(f"\rEpoch: [{epoch}/{self.config['max_epoch']}], Lr: {self.lr:.5f}, Progress: {progress:.2f}%, Loss: {avg_loss:.5f}, ACC: {avg_acc:.2f}% Elapsed time: {time.time() - self.start_time:.2f} seconds.", end='')
        train_loss = loss / total_batches
        train_acc = 100 * top1 / index
        return train_loss, train_acc

    def train_audio_video_non_temporal(self, loader, epoch, total_batches, print_update_interval):
        index, top1, loss = 0, 0, 0
        for num, (audio_feature, visual_feature, labels) in enumerate(loader, start=1):
            self.zero_grad()
            audio_embed = self.model.forward_non_temporal_audio(audio_feature[0].cuda())
            visual_embed = self.model.forward_non_temporal_visual(visual_feature[0].cuda())
            labels = labels[0].reshape((-1)).cuda()
            audio_embed, visual_embed = self.model.forward_cross_attention(audio_embed, visual_embed)
            outsAV= self.model.forward_audio_visual_backend(audio_embed, visual_embed)  
            outsA = self.model.forward_audio_backend(audio_embed)
            outsV = self.model.forward_visual_backend(visual_embed)
            
            # Loss
            nlossAV, _, _, precision = self.lossAV.forward(outsAV, labels)
            nlossA, _, _, _ = self.lossA.forward(outsA, labels)
            nlossV, _, _, _ = self.lossV.forward(outsV, labels)
            nloss = nlossAV + 0.4 * nlossA + 0.4 * nlossV
            loss += nloss.detach().cpu().numpy()
            top1 += precision
            nloss.backward()
            self.optim.step()
            index += len(labels)
            
            # Dynamic output
            progress = 100 * num / total_batches
            avg_loss = loss / num
            avg_acc = 100 * top1 / index
            if num % print_update_interval == 0 or num == total_batches - 1:
                print(f"\rEpoch: [{epoch}/{self.config['max_epoch']}], Lr: {self.lr:.5f}, Progress: {progress:.2f}%, Loss: {avg_loss:.5f}, ACC: {avg_acc:.2f}% Elapsed time: {time.time() - self.start_time:.2f} seconds.", end='')
        train_loss = loss / total_batches
        train_acc = 100 * top1 / index
        return train_loss, train_acc

    
    def train_network(self, loader, epoch, **kwargs):
        use_audio = kwargs.get('use_audio', True)
        use_video = kwargs.get('use_video', True)
        use_temporal = kwargs.get('use_temporal', True)
        self.train()
        self.scheduler.step(epoch - 1)
        
        self.lr = self.optim.param_groups[0]['lr']
        total_batches = loader.__len__()
        print_update_interval = total_batches // 25
        self.start_time = time.time()

        print(f"\rEpoch: [0/{self.config['max_epoch']}], Lr: {self.lr:.5f}, Progress: 0%, Loss: -, ACC: -% Elapsed time: 0 seconds.", end='')
        if use_audio and use_video:
            if use_temporal:
                train_loss, train_acc = self.train_audio_video(loader, epoch, total_batches, print_update_interval)
            else:
                train_loss, train_acc = self.train_audio_video_non_temporal(loader, epoch, total_batches, print_update_interval)
        elif use_audio:
            train_loss, train_acc = self.train_audio(loader, epoch, total_batches, print_update_interval)
        elif use_video:
            train_loss, train_acc = self.train_video(loader, epoch, total_batches, print_update_interval)
        
        # Final message after training completion
        end_time = time.time()
        time_taken = end_time - self.start_time
        print(f"\rCompleted training for epoch {epoch}. Time taken: {time_taken:.2f} seconds, Loss: {train_loss:.5f}, ACC: {train_acc:.2f}%                    ")

        train_metrics = {
            "train_loss": train_loss,
            "learning_rate": self.lr,
            "train_accuracy": train_acc.detach().cpu().item(),
            "train_duration": time_taken,
        }
    
        return train_metrics
    
    
    def eval_audio_video(self, loader, total_batches, print_update_interval):
        pred_scores = {}
        total_loss, top1, index = 0, 0, 0
        with torch.no_grad():  # Ensure no gradients are computed to save memory and computation
            for batch_idx, (video_id, audio_feature, visual_feature, labels) in enumerate(loader):
                
                audio_feature = audio_feature[0].cuda()
                visual_feature = visual_feature[0].cuda()
                labels = labels[0].reshape((-1)).cuda()
                
                audio_embed = self.model.forward_audio_frontend(audio_feature)
                visual_embed = self.model.forward_visual_frontend(visual_feature)
                audio_embed, visual_embed = self.model.forward_cross_attention(audio_embed, visual_embed)
                outsAV = self.model.forward_audio_visual_backend(audio_embed, visual_embed)
                outsA = self.model.forward_audio_backend(audio_embed)
                outsV = self.model.forward_visual_backend(visual_embed)
                
                nlossAV, predScore, _, precision = self.lossAV.forward(outsAV, labels)
                nlossA, _, _, _ = self.lossA.forward(outsA, labels)
                nlossV, _, _, _ = self.lossV.forward(outsV, labels)
                nloss = nlossAV + 0.4 * nlossA + 0.4 * nlossV
                total_loss += nloss.detach().cpu().numpy()
                top1 += precision
                predScore = predScore[:, 1].detach().cpu().numpy()
                video_id = video_id[0]
                if video_id not in pred_scores:
                    pred_scores[video_id] = []
                pred_scores[video_id].extend(predScore)
                index += len(labels)
                # Print progress at the specified interval or if it's the last batch
                if batch_idx % print_update_interval == 0 or batch_idx == total_batches - 1:
                    print(f"\rProcessed {batch_idx+1}/{total_batches} batches. Elapsed time: {time.time() - self.eval_start_time:.2f} seconds.", end='')
        eval_acc = 100 * top1 / index
        eval_loss = total_loss / total_batches

        return pred_scores, eval_loss, eval_acc

    def eval_audio_video_non_temporal(self, loader, total_batches, print_update_interval):
        pred_scores = {}
        total_loss, top1, index = 0, 0, 0
        with torch.no_grad():  # Ensure no gradients are computed to save memory and computation
            for batch_idx, (video_id, audio_feature, visual_feature, labels) in enumerate(loader):
                
                audio_feature = audio_feature[0].cuda()
                visual_feature = visual_feature[0].cuda()
                labels = labels[0].reshape((-1)).cuda()
                
                audio_embed = self.model.forward_non_temporal_audio(audio_feature)
                visual_embed = self.model.forward_non_temporal_visual(visual_feature)
                audio_embed, visual_embed = self.model.forward_cross_attention(audio_embed, visual_embed)
                outsAV = self.model.forward_audio_visual_backend(audio_embed, visual_embed)
                outsA = self.model.forward_audio_backend(audio_embed)
                outsV = self.model.forward_visual_backend(visual_embed)
                
                nlossAV, predScore, _, precision = self.lossAV.forward(outsAV, labels)
                nlossA, _, _, _ = self.lossA.forward(outsA, labels)
                nlossV, _, _, _ = self.lossV.forward(outsV, labels)
                nloss = nlossAV + 0.4 * nlossA + 0.4 * nlossV
                total_loss += nloss.detach().cpu().numpy()
                top1 += precision
                predScore = predScore[:, 1].detach().cpu().numpy()
                video_id = video_id[0]
                if video_id not in pred_scores:
                    pred_scores[video_id] = []
                pred_scores[video_id].extend(predScore)
                index += len(labels)
                # Print progress at the specified interval or if it's the last batch
                if batch_idx % print_update_interval == 0 or batch_idx == total_batches - 1:
                    print(f"\rProcessed {batch_idx+1}/{total_batches} batches. Elapsed time: {time.time() - self.eval_start_time:.2f} seconds.", end='')
        eval_acc = 100 * top1 / index
        eval_loss = total_loss / total_batches

        return pred_scores, eval_loss, eval_acc
    
    def eval_audio(self, loader, total_batches, print_update_interval):
        pred_scores = {}
        total_loss, top1, index = 0, 0, 0
        with torch.no_grad():  # Ensure no gradients are computed to save memory and computation
            for batch_idx, (video_id, audio_feature, _, labels) in enumerate(loader):
                audio_feature = audio_feature[0].cuda()
                labels = labels[0].reshape((-1)).cuda()
                
                audio_embed = self.model.forward_audio_frontend(audio_feature)
                outsA = self.model.forward_audio_backend(audio_embed)

                nlossA, predScore, _, precision = self.lossA.forward(outsA, labels)
                nloss = nlossA
                total_loss += nloss.detach().cpu().numpy()
                top1 += precision
                predScore = predScore[:, 1].detach().cpu().numpy()
                video_id = video_id[0]
                if video_id not in pred_scores:
                    pred_scores[video_id] = []
                pred_scores[video_id].extend(predScore)
                index += len(labels)

                # Print progress at the specified interval or if it's the last batch
                if batch_idx % print_update_interval == 0 or batch_idx == total_batches - 1:
                    print(f"\rProcessed {batch_idx+1}/{total_batches} batches. Elapsed time: {time.time() - self.eval_start_time:.2f} seconds.", end='')
        eval_acc = 100 * top1 / index
        eval_loss = total_loss / total_batches

        return pred_scores, eval_loss, eval_acc
    

    def eval_video(self, loader, total_batches, print_update_interval):
        pred_scores = {}
        total_loss, top1, index = 0, 0, 0
        with torch.no_grad():  # Ensure no gradients are computed to save memory and computation
            for batch_idx, (video_id, _, visual_feature, labels) in enumerate(loader):
                visual_feature = visual_feature[0].cuda()
                labels = labels[0].reshape((-1)).cuda()
                
                visual_embed = self.model.forward_visual_frontend(visual_feature)
                outsV = self.model.forward_visual_backend(visual_embed)

                nlossV, predScore, _, precision = self.lossV.forward(outsV, labels)
                nloss = nlossV
                total_loss += nloss.detach().cpu().numpy()
                top1 += precision
                predScore = predScore[:, 1].detach().cpu().numpy()
                video_id = video_id[0]
                if video_id not in pred_scores:
                    pred_scores[video_id] = []
                pred_scores[video_id].extend(predScore)
                index += len(labels)

                # Print progress at the specified interval or if it's the last batch
                if batch_idx % print_update_interval == 0 or batch_idx == total_batches - 1:
                    print(f"\rProcessed {batch_idx+1}/{total_batches} batches. Elapsed time: {time.time() - self.eval_start_time:.2f} seconds.", end='')
        eval_acc = 100 * top1 / index
        eval_loss = total_loss / total_batches

        return pred_scores, eval_loss, eval_acc
        

    def evaluate_network(self, loader, epoch, eval_csv_save, eval_scores_csv_save, eval_combined, test_combined=None, test_csv_save=None, full_test=False, **kwargs):
        use_audio = kwargs.get('use_audio', True)
        use_video = kwargs.get('use_video', True)
        use_temporal = kwargs.get('use_temporal', True)
        combined_csv = test_combined if full_test else eval_combined
        csv_save_path = test_csv_save if full_test else eval_csv_save
        
        self.eval_start_time = time.time()
        
        total_batches = loader.__len__()
        print_update_interval = total_batches // 25

        self.eval()
        
        print(f"\rProcessed 0/{total_batches} batches. Elapsed time: 0 seconds.", end='')
        if use_audio and use_video:
            if use_temporal:
                pred_scores, eval_loss, eval_acc = self.eval_audio_video(loader, total_batches, print_update_interval)
            else:
                pred_scores, eval_loss, eval_acc = self.eval_audio_video_non_temporal(loader, total_batches, print_update_interval)
        elif use_audio:
            pred_scores, eval_loss, eval_acc = self.eval_audio(loader, total_batches, print_update_interval)
        elif use_video:
            pred_scores, eval_loss, eval_acc = self.eval_video(loader, total_batches, print_update_interval)
        
        eval_duration = time.time() - self.eval_start_time
        
        csv_time_start = time.time()
        # Process CSV for detailed evaluation results
        eval_lines = open(combined_csv).read().splitlines()[1:]
        labels = pd.Series(['SPEAKING_AUDIBLE' for _ in eval_lines])
        # scores = pd.Series(pred_scores)
        eval_res = pd.read_csv(combined_csv)
        # eval_res['score'] = scores
        eval_res['label'] = labels
        eval_res.drop(['label_id'], axis=1, inplace=True)
        # Ensure the 'score' column exists and initialize with NaN or another placeholder value
        eval_res['score'] = np.nan

        for video_id, scores_array in pred_scores.items():
            # Find the first occurrence index for this video_id
            occurrence_indices = eval_res.index[eval_res['video_id'] == video_id].tolist()
            if occurrence_indices:
                eval_res.loc[occurrence_indices[0]:occurrence_indices[-1], 'score'] = scores_array
            else:
                print(f"Warning: Video ID {video_id} not found in evaluation CSV.")


        eval_res.to_csv(csv_save_path, index=False)

        if epoch == 1:
            # If it doesn't exist, rename 'score' column to scores_{epoch} and save
            new_eval_res = eval_res.rename(columns={'score': 'scores_1'})
            new_eval_res.to_csv(eval_scores_csv_save, index=False)
        elif not full_test:
            # Read the history CSV and append new scores as a new column
            scores_history_df = pd.read_csv(eval_scores_csv_save)
            updated_eval_res = pd.read_csv(csv_save_path)

            # Copy the 'score' column from the updated eval_res as 'scores_{epoch}' in the scores_history_df
            scores_history_df[f'scores_{epoch}'] = updated_eval_res['score']
            scores_history_df.to_csv(eval_scores_csv_save, index=False)

        csv_time_duration = time.time() - csv_time_start

        # Execute the external script for mAP calculation
        mAP = 0.0
        proc = subprocess.run([sys.executable, "-O", kwargs.get('mAP_path_AVA'), "-g", combined_csv, "-p", csv_save_path], capture_output=True, text=True)
        output = proc.stdout

        try:
            # Use a regex to find floating point numbers followed by a percentage sign
            match = re.search(r'(\d+\.\d+)%', output)
            if match:
                mAP = float(match.group(1))
            else:
                raise ValueError("mAP value not found in the output.")
        except Exception as e:
            print(f"Error parsing mAP from output: {e}")

        print(f"\rEvaluation completed in {eval_duration:.2f} seconds. Average loss: {eval_loss:.4f}, Precision: {eval_acc:.4f}.")

        # Prepare metrics for logging
        eval_metrics = {
            "mAP": mAP,
            "eval_duration": eval_duration,
            "eval_accuracy": eval_acc.detach().cpu().item(),
            "eval_loss": eval_loss
        }
        
        return eval_metrics

    def save_parameters(self, path):
        torch.save(self.state_dict(), path)

    def load_parameters(self, path):
        selfState = self.state_dict()
        loadedState = torch.load(path)
        for name, param in loadedState.items():
            origName = name
            if name not in selfState:
                name = name.replace("module.", "")
                if name not in selfState:
                    print("%s is not in the model."%origName)
                    continue
            if selfState[name].size() != loadedState[origName].size():
                sys.stderr.write("Wrong parameter length: %s, model: %s, loaded: %s"%(origName, selfState[name].size(), loadedState[origName].size()))
                continue
            selfState[name].copy_(param)



