import torch
from dataclasses import dataclass, field
import math
from typing import List, Optional
from transformers import get_scheduler, AutoModelForCausalLM
from nanovllm import LLM, SamplingParams
import huggingface_hub
import wandb
from simple_grpo.simple_metric import Response, Instance, MathMetric
from simple_grpo.simple_data import MinervaMath, HamishMathORZ
from simple_grpo.grpo_utils import disable_dropout, masked_mean, log_softmax_and_gather, get_train_ds_config, get_eval_ds_config, gradient_checkpointing_enable
import deepspeed
from deepspeed.runtime.pipe.schedule import TrainSchedule
import logging

# Banish the deepspeed logger
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.ERROR)
deepspeed.utils.logging.logger.setLevel(logging.ERROR)

INVALID_LOGPROB = 1.0


@dataclass
class ModelConfig:
    model_name_or_path: str
    model_revision: Optional[str] = None


@dataclass
class TrainConfig:
    """ https://wandb.ai/ai2-llm/open_instruct_internal/runs/09uuc5hk/overview """
    # generation
    rollouts_per_prompt: int = 16
    per_device_train_batch_size: int = 16
    temperature: float = 0.7
    max_response_length: int = 2048

    # trainer
    num_epochs: int = 1
    deepspeed_stage: int = 0
    lr: float = 5e-6 # 5e-7
    lr_scheduler_type: str = 'constant'
    warm_up_steps: int = 0
    num_scheduler_steps: int = 0
    fused_optimizer: bool = False
    clip_lower: float = 0.2
    clip_higher: float = 0.2
    beta: float = 0
    num_mini_batches: int = 2
    masked_mean_axis: Optional[int] = None


@dataclass
class WandbConfig:
    wandb_project_name: str
    wandb_entity: str
    run_name: str
    exp_name: str


@dataclass
class Config:
    train_config: TrainConfig = field(default_factory=TrainConfig)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    wandb_config: Optional[WandbConfig] = None


class WandB:
    def __init__(self, config: Config):
        wandb.init(
            project=config.wandb_config.wandb_project_name,
            entity=config.wandb_config.wandb_entity,
            sync_tensorboard=True,
            config=config,
            name=config.wandb_config.run_name,
            save_code=True,
            tags=[config.wandb_config.exp_name], # TODO: get_wandb_tags()
        )
        self.metrics = {}

    def log_table(self, table: dict[str, List[str]]):
        wandb.log({
            "sample_completions": wandb.Table(
                columns=list(table.keys()),
                data=list(zip(*table.values()))
            )
        })

    def log(self, metric: dict[str, object]):
        self.metrics.update(metric)

    def step(self):
        wandb.log(self.metrics)
        self.metrics = {}


class Trainer:
    def __init__(self, config: Config):
        train_config = config.train_config
        model_config = config.model_config

        self.train_config = train_config

        self.policy_vllm = LLM(
            model_config.model_name_or_path, 
            enforce_eager=True, 
            tensor_parallel_size=1,
            gpu_memory_utilization=0.5 # 0.3
        )
        torch.set_default_device("cuda") # nano-llm resets default device
        self.sampling_params_train = SamplingParams(
            temperature=train_config.temperature, 
            max_tokens=self.train_config.max_response_length
        )
        self.sampling_params_eval  = SamplingParams(
            temperature=train_config.temperature, 
            max_tokens=self.train_config.max_response_length
        )
        
        # @davidh: I would love to use nano-vllm, but it does not have: (1) attention masking or (2) batching
        self.policy: torch.nn.Module = AutoModelForCausalLM.from_pretrained(
            model_config.model_name_or_path,
            revision=model_config.model_revision,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            use_cache=False,
        )
        self.ref_policy: torch.nn.Module = AutoModelForCausalLM.from_pretrained(
            model_config.model_name_or_path,
            revision=model_config.model_revision,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            use_cache=False,
        )

        # TODO: Load existing checkpoint if exists

        disable_dropout(self.policy)
        disable_dropout(self.ref_policy)

        # gradient_checkpointing_enable(self.policy) TODO: Enable gradient ckpting

        optim_params = self.policy.parameters()

        self.optimizer = torch.optim.AdamW(optim_params, lr=train_config.lr, fused=train_config.fused_optimizer)

        scheduler = get_scheduler(
            name=train_config.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=train_config.warm_up_steps,
            num_training_steps=train_config.num_scheduler_steps,
        )

        ds_config = get_train_ds_config(
            offload=False,
            adam_offload=False,
            stage=train_config.deepspeed_stage,
            bf16=True,
        )
        ds_config["train_micro_batch_size_per_gpu"] = train_config.per_device_train_batch_size
        ds_config["gradient_accumulation_steps"] = 1

        self.policy, self.optimizer, _, self.scheduler: TrainSchedule = deepspeed.initialize(
            model=self.policy,
            optimizer=self.optimizer,
            config=ds_config,
            lr_scheduler=scheduler,
            # dist_init_required=True,
        )

        ds_config = get_eval_ds_config(
            offload=False,
            # inference model only has stage 3 (sharding) or stage 0 (no sharding)
            # stage 2 is optimizer sharding which doesn't apply to inference
            stage=train_config.deepspeed_stage if train_config.deepspeed_stage == 3 else 0,
            bf16=True,
        )
        ds_config["train_micro_batch_size_per_gpu"] = train_config.per_device_train_batch_size
        ds_config["gradient_accumulation_steps"] = 1
        self.ref_policy, *_ = deepspeed.initialize(model=self.ref_policy, config=ds_config)

        self.ref_policy.eval()
        self.policy.train()

        self.wandb = WandB(config)

    def forward(
        self,
        model: AutoModelForCausalLM,
        query_response: torch.LongTensor,
        attention_mask: torch.LongTensor,
        position_ids: torch.LongTensor,
        pad_token_id: int,
        temperature: float,
    ) -> torch.Tensor:
        # Replace pad tokens with 0s so that we don't run into index out of bounds errors
        padding_mask = query_response != pad_token_id
        input_ids = torch.masked_fill(query_response, ~padding_mask, 0)
        # NOTE: the [:-1] and [1:] are because the logits and generated tokens are off by 1 in index
        output = model(
            input_ids=input_ids[:, :-1],
            # @vwxyzjn: without clamp, we get index out of bounds errors
            attention_mask=attention_mask[:, :-1].clamp(0, 1),
            positions=position_ids[:, :-1],
        )
        logits = output.logits
        logits /= temperature + 1e-7
        logprob = log_softmax_and_gather(logits, input_ids[:, 1:])
        return logprob
    

    def broadcast_to_vllm(self):
        policy_unwrapped = self.policy.module if hasattr(self.policy, "module") else self.policy # unwrap DeepSpeed

        # Copy over policy weights to vLLM
        # for name, param in policy_unwrapped.named_parameters():
        #     self.policy_vllm.model_runner.model.get_parameter(name).data.copy_(param.data)
        from nanovllm.utils.loader import copy_weights
        copy_weights(policy_unwrapped, self.policy_vllm.model_runner.model)

        # Clear vLLM cache
        for seq in self.policy_vllm.scheduler.running:
            self.policy_vllm.scheduler.block_manager.deallocate(seq)

        for seq in self.policy_vllm.scheduler.waiting:
            if seq.block_table:
                self.policy_vllm.scheduler.block_manager.deallocate(seq)
        torch.cuda.empty_cache()


    def update_ref_policy(self):
        policy_unwrapped = self.policy.module if hasattr(self.policy, "module") else self.policy # unwrap DeepSpeed
        ref_policy_unwrapped = self.ref_policy.module if hasattr(self.ref_policy, "module") else self.ref_policy # unwrap DeepSpeed

        # Copy over policy weights to reference policy
        for name, param in policy_unwrapped.named_parameters():
            ref_policy_unwrapped.get_parameter(name).data.copy_(param.data)
        torch.cuda.empty_cache()
    

    def update_model(
        self,
        responses,
        attention_masks,
        position_ids,
        advantages,
        response_masks,
        pad_token_id: int,
        temperature: float,
        num_mini_batches: int,
    ):
        accumulation_steps = math.ceil(len(responses) / num_mini_batches - 0.5)

        with torch.no_grad():
            # Compute logprob of reference policy
            ref_logprobs = []
            for i in range(len(responses)):
                response_mask = response_masks[i]
                ref_logprob = self.forward(
                    self.ref_policy,
                    responses[i],
                    attention_masks[i],
                    position_ids[i],
                    pad_token_id,
                    temperature,
                )
                response_mask = response_mask.bool()
                ref_logprob = torch.masked_fill(
                    ref_logprob, ~response_mask[:, 1:], INVALID_LOGPROB
                )
                ref_logprobs.append(ref_logprob)
                torch.cuda.empty_cache()

        local_step = 0
        old_logprobs = [None for _ in range(len(responses))]
        kl_stats = torch.zeros(len(responses))
        # kl_loss_stats = torch.zeros(len(responses))
        # pg_clipfrac_stats = torch.zeros(len(responses))
        # pg_loss_stats = torch.zeros(len(responses))
        # loss_stats = torch.zeros(len(responses))
        # ratio_stats = torch.zeros(len(responses))
        for epoch_idx in range(self.train_config.num_epochs):
            for i in range(len(responses)):
                mb_ref_logprob = ref_logprobs[i]
                mb_advantages = advantages[i]
                mb_response_masks = response_masks[i]
                mb_response_masks_bool = mb_response_masks[:, 1:].bool()

                mb_new_logprobs = self.forward(
                    self.policy,
                    responses[i],
                    attention_masks[i],
                    position_ids[i],
                    pad_token_id,
                    temperature,
                )
                mb_new_logprobs = torch.masked_fill(mb_new_logprobs, ~mb_response_masks_bool, INVALID_LOGPROB)

                # Cache the old logprobs
                with torch.no_grad():
                    if epoch_idx == 0:
                        old_logprobs[i] = mb_new_logprobs
                    mb_old_logprobs = old_logprobs[i].detach()

                # Calculate the policy's loss
                logprobs_diff = mb_new_logprobs - mb_old_logprobs
                ratio = torch.exp(logprobs_diff)
                pg_losses = -mb_advantages[:, 1:] * ratio
                pg_losses2 = -mb_advantages[:, 1:] * torch.clamp(
                    ratio, 1.0 - self.train_config.clip_lower, 1.0 + self.train_config.clip_higher
                )
                pg_loss_max = torch.max(pg_losses, pg_losses2)

                # Here we recalculate kl: we want the KL loss to backpropagate through the model
                # We also clamp the KL loss to avoid numerical instability
                # https://chatgpt.com/share/679d0ed9-8f48-8011-926e-e274b15ae8ae
                ref_logprobs_diff = (mb_new_logprobs - mb_ref_logprob).clamp(-40.0, 40.0)
                # numerically stable kl (kl3 in open-instruct)
                kl = torch.expm1(-ref_logprobs_diff) + ref_logprobs_diff  

                # grpo change: directly subtract KL in loss (add)
                loss = masked_mean(
                    pg_loss_max + (self.train_config.beta * kl), 
                    mb_response_masks_bool, self.train_config.masked_mean_axis
                )
                loss = loss / accumulation_steps
                self.policy.backward(loss)
                if (local_step + 1) % accumulation_steps == 0:
                    self.policy.step()
                local_step += 1
                with torch.no_grad():
                    # TODO: We use kl 3, but should also track kl 1

                    # NOTE: in packed implementation, kl calculation are averages over response tokens
                    kl_stats[i] = masked_mean(kl, mb_response_masks_bool, self.train_config.masked_mean_axis).float()
                    # kl_loss_stats[i] = kl_stats[i] * self.train_config.beta
                    # pg_clipfrac_stats[i] = masked_mean(
                    #     (pg_losses2 > pg_losses).float(), mb_response_masks_bool, self.train_config.masked_mean_axis
                    # )
                    # pg_loss_stats[i] = masked_mean(pg_loss_max, mb_response_masks_bool, self.train_config.masked_mean_axis)
                    # loss_stats[i] = loss
                    # ratio_stats[i] = masked_mean(ratio, mb_response_masks_bool, self.train_config.masked_mean_axis)

        # with torch.no_grad():
        #     self.local_metrics.add("objective/kl_avg", kl_stats.mean())
        #     self.local_metrics.add("loss/policy_avg", pg_loss_stats.mean())
        #     self.local_metrics.add("loss/kl_avg", kl_loss_stats.mean())
        #     self.local_metrics.add("loss/total_avg", loss_stats.mean())
        #     self.local_metrics.add("policy/clipfrac_avg", pg_clipfrac_stats.mean())
        #     self.local_metrics.add("val/ratio", ratio_stats.mean())
        #     self.local_metrics.add("val/ratio_var", ratio_stats.var())
        #     self.local_metrics.add("lr", self.scheduler.get_last_lr()[0])
        #     return self.local_metrics.get_metrics_list()

        self.wandb.log({'train/kl_avg': kl_stats.mean()})
        self.wandb.log({'lr': self.scheduler.get_last_lr()[0]})

        return kl_stats.mean()


    def generate(self, rollout_prompts: List[str]) -> List[str]:
        # Generate rollouts
        rollouts = self.policy_vllm.generate(
            prompts=rollout_prompts, 
            sampling_params=self.sampling_params_train,
            use_tqdm=False
        )
        rollouts = [out["text"] for out in rollouts]

        # # TODO: Generate evaluations
        # evals = self.policy_vllm.generate(
        #     prompts=eval_prompts, 
        #     sampling_params=self.sampling_params_eval
        # )
        # evals = [out["text"] for out in evals]

        return rollouts

    
    def batch_generate(self, batch_prompts: List[Instance], rollouts_per_prompt: int) -> List[List[Response]]:
        # (prompts * rollouts_per_prompt)
        batch_rollout_instances = []
        batch_rollout_requests = []
        for prompt in batch_prompts:
            batch_rollout_instances.extend([prompt] * rollouts_per_prompt)
            batch_rollout_requests.extend([prompt.request] * rollouts_per_prompt)

        batch_rollouts = self.generate(batch_rollout_requests)

        # (prompts * rollouts_per_prompt)
        batch_responses = []
        for instance, rollout in zip(batch_rollout_instances, batch_rollouts):
            batch_responses.append(Response(input=instance, output=rollout))
        
        # (prompts, rollouts_per_prompt)
        batch_responses = [batch_responses[i:i+rollouts_per_prompt] for i in range(0, len(batch_responses), rollouts_per_prompt)]

        return batch_responses


    def reward(self, responses: List[Response]) -> torch.Tensor:
        metric = MathMetric(responses)
        metric.grade_responses()
        scores = metric.scores
        scores = torch.tensor(scores, dtype=torch.float32)

        # TODO: format reward
        
        return scores

    def prepare_forward_pass(self, batch_advantages: List[torch.Tensor], batch_responses: List[List[Response]]):
        #####################
        # Prepare outputs for forward pass (I wrote all this, not sure if it's correct)
        #####################
        all_rollouts_ids = []
        all_attention_masks = []
        all_position_ids = []
        all_advantages = []
        all_response_masks = []
        for advantages, responses in zip(batch_advantages, batch_responses):
            rollout_prompts = [response.input.request for response in responses]
            rollouts = [response.output for response in responses]

            padding_side = "left" # for Qwen 3

            # First tokenize just the prompts to get their lengths
            prompt_token_lengths = [
                len(self.policy_vllm.tokenizer.encode(prompt)) 
                for prompt in rollout_prompts
            ]

            # Get tokenizer outputs for prompts + rollouts
            tokenizer_outputs = self.policy_vllm.tokenizer(
                [prompt + rollout for prompt, rollout in zip(rollout_prompts, rollouts)],
                padding=True,
                truncation=True,
                max_length=self.train_config.max_response_length,
                return_tensors="pt",
                padding_side=padding_side
            )

            rollouts_ids = tokenizer_outputs["input_ids"]
            attention_masks = tokenizer_outputs["attention_mask"]
            position_ids = torch.arange(rollouts_ids.shape[1], dtype=torch.long)[None, :].repeat(rollouts_ids.shape[0], 1)
            pad_token_id = self.policy_vllm.tokenizer.pad_token_id
            
            # Create response masks - 1 for response tokens, 0 for prompt tokens
            response_masks = torch.zeros_like(rollouts_ids)
            for i, prompt_len in enumerate(prompt_token_lengths):
                if padding_side == "left":
                    # For left padding, response starts after prompt from the right
                    response_start = rollouts_ids.shape[1] - (rollouts_ids[i] != pad_token_id).sum() + prompt_len
                    response_masks[i, response_start:] = 1
                else:
                    response_masks[i, prompt_len:] = 1

            # token-level advantages
            advantages = advantages.unsqueeze(1).expand(-1, rollouts_ids.shape[1]) # (batch, seq_len)

            # Collect tensors for batching
            all_rollouts_ids.extend(rollouts_ids)
            all_attention_masks.extend(attention_masks)
            all_position_ids.extend(position_ids)
            all_advantages.extend(advantages)
            all_response_masks.extend(response_masks)
    
        #####################

        # Split updates into micro-batches
        mb_rollouts_ids = []
        mb_attention_masks = []
        mb_position_ids = []
        mb_advantages = []
        mb_response_masks = []
        for i in range(0, len(all_rollouts_ids), self.train_config.num_mini_batches):
            batch_end = min(i + self.train_config.num_mini_batches, len(all_rollouts_ids))
            mb_rollouts_ids.append(all_rollouts_ids[i:batch_end])
            mb_attention_masks.append(all_attention_masks[i:batch_end])
            mb_position_ids.append(all_position_ids[i:batch_end])
            mb_advantages.append(all_advantages[i:batch_end])
            mb_response_masks.append(all_response_masks[i:batch_end])
        mb_rollouts_ids = [torch.stack(batch, dim=0) for batch in mb_rollouts_ids]
        mb_attention_masks = [torch.stack(batch, dim=0) for batch in mb_attention_masks]
        mb_position_ids = [torch.stack(batch, dim=0) for batch in mb_position_ids]
        mb_advantages = [torch.stack(batch, dim=0) for batch in mb_advantages]
        mb_response_masks = [torch.stack(batch, dim=0) for batch in mb_response_masks]

        return (
            mb_rollouts_ids,
            mb_attention_masks,
            mb_position_ids,
            mb_advantages,
            mb_response_masks,
        )


    def train_step(self, batch_prompts: List[Instance]):
        # Generate responses
        batch_responses: List[List[Response]] = self.batch_generate(
            batch_prompts=batch_prompts,
            rollouts_per_prompt=self.train_config.rollouts_per_prompt
        ) # (prompts, rollouts_per_prompt)

        batch_scores = []
        batch_advantages = []
        responses: List[Response]
        for responses in batch_responses:
            # Compute rewards
            scores = self.reward(responses) # (responses,)
            batch_scores += scores

            # Compute GRPO
            scores_per_prompt = scores.reshape(-1, self.train_config.rollouts_per_prompt)
            mean_group_reward = torch.mean(scores_per_prompt, dim=-1)
            std_group_reward  = torch.std(scores_per_prompt, dim=-1)
            mean_group_reward = mean_group_reward.repeat_interleave(self.train_config.rollouts_per_prompt)
            std_group_reward  = std_group_reward.repeat_interleave(self.train_config.rollouts_per_prompt)
            advantages = (scores - mean_group_reward) / (std_group_reward + 1e-8) # standard normalization

            # In GRPO, if the std of grouped rewards is 0, then there is zero gradient for the batch
            # of args.num_samples_per_prompt_rollout responses, so we need to filter out those batches
            if std_group_reward == 0:
                continue
            
            batch_advantages += [advantages]

        self.wandb.log({"train/verifiable_correct_rate": sum(batch_scores) / len(batch_scores)})
        self.wandb.log_table({"sample_completions": batch_responses[0]}) # log first batch of completions # TODO: Get a table of prompt, response, scores, ground_truth

        # Tokenize + get attention mask
        mb_rollouts_ids, mb_attention_masks, mb_position_ids, \
            mb_advantages, mb_response_masks = self.prepare_forward_pass(
            batch_advantages = batch_advantages,
            batch_responses = batch_responses
        )
        
        # Gradient step
        self.update_model(
            mb_rollouts_ids,
            mb_attention_masks,
            mb_position_ids,
            mb_advantages,
            mb_response_masks,
            self.policy_vllm.tokenizer.pad_token_id,
            self.train_config.temperature,
            self.train_config.num_mini_batches,
        )

        self.broadcast_to_vllm()

        self.wandb.step()

    
    def train(self):
        # dataset = HamishMathORZ()
        dataset = MinervaMath("algebra") # TODO: Build an actual dataloader (with shuffling, etc.)
        instances: List[Instance] = dataset.requests
        
        for i in range(0, len(instances), self.train_config.per_device_train_batch_size):
            batch_end = min(i + self.train_config.per_device_train_batch_size, len(instances))
            prompts = instances[i:batch_end]

            self.train_step(prompts)


def main():
    torch.cuda.set_device('cuda:0')

    model_name = "Qwen/Qwen3-0.6B"
    # model_name = "Qwen/Qwen3-32B"
    model_path = huggingface_hub.snapshot_download(model_name)

    trainer = Trainer(
        config = Config(
            model_config=ModelConfig(
                model_name_or_path=model_path
            ),
            train_config=TrainConfig(),
            wandb_config=WandbConfig(
                wandb_project_name="ai2-llm",
                wandb_entity="simple-trainer",
                run_name="debug_runs",
                exp_name="debug"
            )
        )
    )

    trainer.train()


if __name__ == '__main__': main()