import os
import deepspeed
import torch
from torch.utils.data import DataLoader
from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
)
from tqdm import tqdm 
from torch.utils.data.distributed import DistributedSampler
import torch.distributed.tensor.parallel as tp
import wandb
from sherlock_dataset import SherlockDataset, SherlockDataCollator
from loss_utils import get_logprobs, dpo_loss
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--beta', type=float, default=0.5, help='Beta parameter for DPO')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                       help='Label smoothing value for DPO')
    parser.add_argument('--dataloader_num_workers', type=int, default=16, help='Beta parameter for DPO')
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs for DPO')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for deepspeed')
    parser.add_argument('--num_gpus', type=int, default=2, help='local rank for deepspeed')
    parser.add_argument('--save_dir', type=str, default="./", help='path to save the trained checkpoints')
    parser.add_argument('--save_interval', type=int, default=50, help='save frequency for checkpoints')
    parser.add_argument('--data_dir', type=str, default="./", help='Root directory containing data files')
    parser.add_argument('--json_file', type=str, default="sherlock_train_v1_1.json", help='name of the json file containing the dataset')
    parser.add_argument('--pref_file', type=str, default="sherlock_train_preference_data.jsonl", help='name of the \
                        json file containing preference dataset for DPO keys: accepted and rejected')
    parser.add_argument('--ref_accept_lps', type=str, default="sherlock_train_ref_accept_logpbs.npy", help='file name containing \
                        pre-computed log probabilities for accepted completions')
    parser.add_argument('--ref_reject_lps', type=str, default="sherlock_train_ref_reject_logpbs.npy", help='file name containing \
                        pre-computed log probabilities for rejected completions')

    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()

def main():
    args = parse_args()
 
    model_id = "google/paligemma2-10b-ft-docci-448"
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16,
    ).train()
    
    processor = PaliGemmaProcessor.from_pretrained(model_id)
    for param in model.vision_tower.parameters():
        param.requires_grad = False

    for param in model.multi_modal_projector.parameters():
        param.requires_grad = False

    # checkpointing activations because those are very large for this model    
    model.gradient_checkpointing_enable()
    model_engine, optimizer, _, _ = deepspeed.initialize(model=model, 
                                                         config=args.deepspeed_config)
    rank = deepspeed.comm.get_rank()
    wandb.init(
        project="pali-gemma-dpo",  # Your project name
        config={
            "model": "paligemma2-10b-ft-docci-448",
            "batch_size": model_engine.train_micro_batch_size_per_gpu(),
            "grad_accum_steps": model_engine.gradient_accumulation_steps(),
            "label_smoothing": args.label_smoothing,
        }
    )
   
    # Initialize dataset
    dataset = SherlockDataset(
        base_dir=args.data_dir, 
        json_file=args.json_file,
        pref_file=args.pref_file,
        ref_accept_logprobs=args.ref_accept_lps,
        ref_reject_logprobs=args.ref_reject_lps,
    )

    sampler = DistributedSampler(
            dataset,
            shuffle=True,
            drop_last=True,
            num_replicas=deepspeed.comm.get_world_size(),
            rank=rank,
        )
    
    batch_size = model_engine.train_micro_batch_size_per_gpu()
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=SherlockDataCollator(processor),
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        sampler=sampler
    )
    
    num_epochs = args.num_epochs
    grad_accum_steps = model_engine.gradient_accumulation_steps()

    checkpoint_dir = args.save_dir
    best_reward = -float('inf')
    save_interval = args.save_interval

    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)

    for e in range(num_epochs):
        progress_bar = tqdm(
            total=len(dataloader),
            disable=rank != 0,
            desc=f"Processing on {deepspeed.comm.get_world_size()} GPUs"
            )
        
        for batch_idx, batch in enumerate(dataloader):

            global_batch = e * len(dataloader) + batch_idx

            ref_accept_lps = batch.pop("ref_accept_lps")
            ref_reject_lps = batch.pop("ref_reject_lps")

            batch = {k: v.to(rank) for k, v in batch["tokens"].items()}
            outputs = model_engine(**batch)
            
            # computing dpo_loss
            labels = batch["labels"][:, 1:].clone()
            logits = outputs["logits"][:, :-1, :].contiguous()
            policy_lps, _ = get_logprobs(labels, logits)
            policy_lps = policy_lps.to(rank)

            ref_accept_lps = ref_accept_lps.to(policy_lps.device)
            ref_reject_lps = ref_reject_lps.to(policy_lps.device)

            policy_accept_lps = policy_lps[:len(ref_accept_lps)].contiguous()
            policy_reject_lps = policy_lps[len(ref_accept_lps):].contiguous()

            loss, chosen_reward, reject_reward = dpo_loss(policy_accept_lps, policy_reject_lps, ref_accept_lps, ref_reject_lps)
            
            model_engine.backward(loss)
            # calc grad norm
            total_norm = model_engine.get_global_grad_norm()

            model_engine.step()

            if rank == 0:
                progress_bar.update()
    
            if (1 + global_batch) % grad_accum_steps == 0:

                current_lr = model_engine.optimizer.param_groups[0]['lr']
                current_margin = chosen_reward - reject_reward
                torch.distributed.all_reduce(current_margin, op=torch.distributed.ReduceOp.SUM)
                current_margin = current_margin.item()
                # Update best reward and save checkpoint
                if current_margin > best_reward:
                    best_reward = current_margin
                    model_engine.save_checkpoint(
                        save_dir=checkpoint_dir,
                        tag=f"best_reward",
                        client_state={"best_reward": best_reward}
                    )
                    print(f"New best reward margin: {best_reward:.4f}. Checkpoint saved.")

                print(f"RANK: {rank}, Loss: {loss.item()}, Reward Margin: {current_margin:.4f}, LR: {current_lr} Grad Norm: {total_norm}")
                
                wandb.log({
                    "loss": loss.item(),
                    "reward_margin": current_margin,
                    "learning_rate": current_lr,
                    "best_reward": best_reward,
                    "grad_norm": total_norm,
                }, step=(global_batch+1) // grad_accum_steps)

                # Save periodic checkpoint every 50 update steps
                update_step = (1 + global_batch) // grad_accum_steps
                if update_step % save_interval == 0:
                    model_engine.save_checkpoint(
                        save_dir=checkpoint_dir,
                        tag=f"step_{global_batch + 1}",
                        client_state={"step": global_batch + 1}
                    )
                    print(f"Saved periodic checkpoint at step {global_batch + 1}")
                
    # Save final checkpoint at end of training
    if rank == 0:
        model_engine.save_checkpoint(
            save_dir=checkpoint_dir,
            tag="final",
            client_state={"final_step": global_batch}
        )

if __name__ == "__main__":
    main()
