########## The following part is copied from Transformers' trainer (3.4.0) ##########

# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the spe                                                                                                                                                                                                                                                                                               cific language governing permissions and
# limitations under the License.
"""
The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
"""
import gc
import collections
import inspect
import math
import os
import pdb
import re
import copy
import time
import shutil
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

from torch.utils.tensorboard import SummaryWriter 
import numpy as np
import torch
from packaging import version
from torch import nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from models import LMForPromptFinetuning, BertForPromptFinetuning, RobertaForPromptFinetuning, DebertaForPromptFinetuning, resize_token_type_embeddings
import higher
import deepspeed
import transformers
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from prototype_loss import prototypical_loss as loss_fn,euclidean_dist
from proto_sampler import PrototypicalBatchSampler
if 1:
    from peft import (
        get_peft_config,
        get_peft_model,
        get_peft_model_state_dict,
        set_peft_model_state_dict,
        LoraConfig,
        PeftType,
        PrefixTuningConfig,
        PromptEncoderConfig,
    )
from transformers.file_utils import (
    CONFIG_NAME,
    WEIGHTS_NAME,
    PushToHubMixin,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
    is_training_run_on_sagemaker,
)
# from transformers.integrations import (
#     default_hp_search_backend,
#     is_comet_available,
#     is_optuna_available,
#     is_ray_available,
#     is_tensorboard_available,
#     is_wandb_available,
#     run_hp_search_optuna,
#     run_hp_search_ray,
#     is_fairscale_available,
#     deepspeed_init,
#     is_deepspeed_zero3_enabled,
# )
import datasets
from transformers.integrations import (
    default_hp_search_backend,
    is_comet_available,
    is_optuna_available,
    is_ray_available,
    is_tensorboard_available,
    is_wandb_available,
    run_hp_search_optuna,
    run_hp_search_ray,
    is_fairscale_available,
)
global wxytest
wxytest=0
# global protonet
# protonet=1
from transformers.models.auto.modeling_auto import MODEL_FOR_QUESTION_ANSWERING_MAPPING
from transformers.modeling_utils import PreTrainedModel
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    DistributedTensorGatherer,
    IterableDatasetShard,
    SequentialDistributedSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    get_tpu_sampler,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalPrediction,
    HPSearchBackend,
    PredictionOutput,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    default_hp_space,
    speed_metrics,
    set_seed,
)
from transformers.training_args import TrainingArguments
from transformers.utils import logging
from dataset import patch_data
from tqdm import tqdm, trange

_use_native_amp = False
_use_apex = False
#
# DEFAULT_CALLBACKS = [DefaultFlowCallback]
# DEFAULT_PROGRESS_CALLBACK = ProgressCallback
#
# if is_in_notebook():
#     from transformers.utils.notebook import NotebookProgressCallback
#
#     DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback
#
# # Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transformers.file_utils import is_apex_available

    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast

_is_torch_generator_available = False
_is_native_amp_available = False

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from .utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if 0 and is_apex_available():
    from apex import amp

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_torch_generator_available = True
    _is_native_amp_available = True
    from torch.cuda.amp import autocast

# if is_datasets_available():
#     import datasets

# if is_torch_tpu_available():
#     import torch_xla.core.xla_model as xm
#     import torch_xla.debug.metrics as met
#     import torch_xla.distributed.parallel_loader as pl

# if is_fairscale_available():
#     dep_version_check("fairscale")
#     import fairscale
#     from fairscale.nn.data_parallel import FullyShardedDataParallel as FullyShardedDDP
#     from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
#     from fairscale.nn.wrap import auto_wrap
#     from fairscale.optim import OSS
#     from fairscale.optim.grad_scaler import ShardedGradScaler

if is_sagemaker_dp_enabled():
    import smdistributed.dataparallel.torch.distributed as dist
    from smdistributed.dataparallel.torch.parallel.distributed import DistributedDataParallel as DDP
else:
    import torch.distributed as dist

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp

    from .trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat

# if is_training_run_on_sagemaker():
#     logging.add_handler(StreamHandler(sys.stdout))

if TYPE_CHECKING:
    import optuna
from accelerate import Accelerator

logger = logging.get_logger(__name__)


########## The above part is copied from Transformers' trainer (3.4.0) ##########

def default_dev_objective(metrics):
    """
    Objective used for picking the best model on development sets
    """
    if "eval_mnli/acc" in metrics:
        return metrics["eval_mnli/acc"]
    elif "eval_mnli-mm/acc" in metrics:
        return metrics["eval_mnli-mm/acc"]
    elif "eval_f1" in metrics:
        return metrics["eval_f1"]
    elif "eval_mcc" in metrics:
        return metrics["eval_mcc"]
    elif "eval_pearson" in metrics:
        return metrics["eval_pearson"]
    elif "eval_acc" in metrics:
        return metrics["eval_acc"]

    raise Exception("No metric founded for {}".format(metrics))


class Trainer(transformers.Trainer):
    """
    Adding some functions based on Transformers' Trainer class.
    """

    def __init__(
            self,
            model: Union[PreTrainedModel, torch.nn.Module] = None,
            another_model: Union[PreTrainedModel, torch.nn.Module] = None,
            args: TrainingArguments = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            train_demo_dataset: Optional[Dataset] = None,
            another_train_dataset:Optional[Dataset] = None,
            un_train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            test_dataset: Optional[Dataset] = None,
            tokenizer: Optional["PreTrainedTokenizerBase"] = None,
            model_init: Callable[[], PreTrainedModel] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
            **kwargs,
    ):
        super().__init__(model=model, args=args, data_collator=data_collator, train_dataset=train_dataset, eval_dataset=eval_dataset)
        if args is None:
            logger.info("No `TrainingArguments` passed, using the current path as `output_dir`.")
            args = TrainingArguments("tmp_trainer")
        self.args = args

        # Seed must be set before instantiating the model when using model
        set_seed(self.args.seed)

        self.hp_name = None
        self.deepspeed = None
        self.is_in_train = False
        # Mixed precision setup
        self.use_apex = False
        self.use_amp = False
        self.fp16_backend = None
        self.current_flos = 0
        self._total_loss_scalar = 0.0
        self._memory_tracker = TrainerMemoryTracker(self.args.skip_memory_metrics)
        self._memory_tracker.start()
        self.test_dataset=test_dataset
        self.accelerator = Accelerator(cpu=self.args.cpu)
        self.another_train_dataset=another_train_dataset

        assert (
                model is not None or model_init is not None
        ), "You must provide a model to use `Trainer`, either by using the `model` argument or the `model_init` argument."
        self.model_init = model_init
        if model is None and model_init is not None:
            model = self.call_model_init()
        #self.model = model.to(args.device) if model is not None else None
        #new add
        self.model = model.to(self.accelerator.device) if model is not None else None
        self.another_model=another_model.to(self.accelerator.device) if model is not None else None
        self.model_list=[]
        self.optimizer_list=[]
        self.lr_scheduler_list=[]
        default_collator = default_data_collator if tokenizer is None else DataCollatorWithPadding(tokenizer)
        self.data_collator = data_collator if data_collator is not None else default_collator
        self.train_dataset = train_dataset
        self.train_demo_dataset = train_demo_dataset
        self.un_train_dataset = un_train_dataset

        self.eval_dataset = eval_dataset

        self.tokenizer = tokenizer

        self.sharded_ddp = None
        if len(args.sharded_ddp) > 0:
            if args.deepspeed:
                raise ValueError(
                    "Using --sharded_ddp xxx together with --deepspeed is not possible, deactivate one of those flags."
                )

            if args.local_rank == -1:
                raise ValueError("Using sharded DDP only works in distributed training.")
            elif not is_fairscale_available():
                raise ImportError("Sharded DDP training requires fairscale: `pip install fairscale`.")
            elif ShardedDDPOption.SIMPLE not in args.sharded_ddp and FullyShardedDDP is None:
                raise ImportError(
                    "Sharded DDP in a mode other than simple training requires fairscale version >= 0.3, found "
                    f"{fairscale.__version__}. Upgrade your fairscale library: `pip install --upgrade fairscale`."
                )
            elif ShardedDDPOption.SIMPLE in args.sharded_ddp:
                self.sharded_ddp = ShardedDDPOption.SIMPLE
            elif ShardedDDPOption.ZERO_DP_2 in args.sharded_ddp:
                self.sharded_ddp = ShardedDDPOption.ZERO_DP_2
            elif ShardedDDPOption.ZERO_DP_3 in args.sharded_ddp:
                self.sharded_ddp = ShardedDDPOption.ZERO_DP_3

        self.compute_metrics = compute_metrics
        self.optimizer, self.lr_scheduler = optimizers
        

        if model_init is not None and (self.optimizer is not None or self.lr_scheduler is not None):
            raise RuntimeError(
                "Passing a `model_init` is incompatible with providing the `optimizers` argument."
                "You should subclass `Trainer` and override the `create_optimizer_and_scheduler` method."
            )
        callbacks = DEFAULT_CALLBACKS if callbacks is None else DEFAULT_CALLBACKS + callbacks

        self.callback_handler = CallbackHandler(
            callbacks, self.model, self.tokenizer, self.optimizer, self.lr_scheduler
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)

        # Deprecated arguments
        if "tb_writer" in kwargs:
            warnings.warn(
                "Passing `tb_writer` as a keyword argument is deprecated and won't be possible in a "
                + "future version. Use `TensorBoardCallback(tb_writer=...)` instead and pass it to the `callbacks`"
                + "argument",
                FutureWarning,
            )
            tb_writer = kwargs.pop("tb_writer")
            self.remove_callback(TensorBoardCallback)
            self.add_callback(TensorBoardCallback(tb_writer=tb_writer))
        if "prediction_loss_only" in kwargs:
            warnings.warn(
                "Passing `prediction_loss_only` as a keyword argument is deprecated and won't be possible in a "
                + "future version. Use `args.prediction_loss_only` instead. Setting "
                + f"`args.prediction_loss_only={kwargs['prediction_loss_only']}",
                FutureWarning,
            )
            self.args.prediction_loss_only = kwargs.pop("prediction_loss_only")
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

        # Will be set to True by `self._setup_loggers()` on first call to `self.log()`.
        self._loggers_initialized = False

        # Create output directory if needed
        if self.is_world_process_zero():
            os.makedirs(self.args.output_dir, exist_ok=True)
        if is_torch_tpu_available() and isinstance(self.model, PreTrainedModel):
            # Set an xla_device flag on the model's config.
            # We'll find a more elegant and not need to do this in the future.
            self.model.config.xla_device = True
        if not callable(self.data_collator) and callable(getattr(self.data_collator, "collate_batch", None)):
            self.data_collator = self.data_collator.collate_batch
            warnings.warn(
                (
                        "The `data_collator` should now be a simple callable (function, class with `__call__`), classes "
                        + "with a `collate_batch` are deprecated and won't be supported in a future version."
                ),
                FutureWarning,
            )

        if args.max_steps > 0:
            logger.info("max_steps is given, it will override any value given in num_train_epochs")

        # Enforce rules on using datasets with no __len__
        if train_dataset is not None and not isinstance(train_dataset, collections.abc.Sized) and args.max_steps <= 0:
            raise ValueError("train_dataset does not implement __len__, max_steps has to be specified")
        if un_train_dataset is not None and not isinstance(un_train_dataset,
                                                           collections.abc.Sized) and args.max_steps <= 0:
            raise ValueError("unlabeled train_dataset does not implement __len__, max_steps has to be specified")
        if eval_dataset is not None and not isinstance(eval_dataset, collections.abc.Sized):
            raise ValueError("eval_dataset must implement __len__")

        if is_datasets_available():
            if isinstance(train_dataset, datasets.Dataset):
                self._remove_unused_columns(self.train_dataset, description="training")
            if isinstance(un_train_dataset, datasets.Dataset):
                self._remove_unused_columns(self.un_train_dataset, description="un_training")
            if isinstance(eval_dataset, datasets.Dataset):
                self._remove_unused_columns(self.eval_dataset, description="evaluation")

        self.state = TrainerState()
        self.control = TrainerControl()
        # Internal variable for total_flos used to count as tensors (for distributed + TPU), will be sent in the
        # state at each call to self.log.
        self._total_flos = None
        if self.args.fp16 and _use_native_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        self.hp_search_backend = None
        self.use_tune_checkpoints = False
        default_label_names = (
            ["start_positions, end_positions"]
            if type(self.model) in MODEL_FOR_QUESTION_ANSWERING_MAPPING.values()
            else ["labels"]
        )
        self.label_names = default_label_names if self.args.label_names is None else self.args.label_names
        self.control = self.callback_handler.on_init_end(self.args, self.state, self.control)

        if args.fp16:
            if args.fp16_backend == "auto":
                self.fp16_backend = "amp" if _is_native_amp_available else "apex"
            else:
                self.fp16_backend = args.fp16_backend
            logger.info(f"Using {self.fp16_backend} fp16 backend")

        if args.fp16 and not args.deepspeed:  # deepspeed manages its own fp16
            if self.fp16_backend == "amp":
                self.use_amp = True
                if is_sagemaker_mp_enabled():
                    self.scaler = smp.amp.GradScaler()
                elif self.sharded_ddp is not None:
                    self.scaler = ShardedGradScaler()
                else:
                    self.scaler = torch.cuda.amp.GradScaler()
            else:
                if not is_apex_available():
                    raise ImportError(
                        "Using FP16 with APEX but APEX is not installed, please refer to https://www.github.com/nvidia/apex."
                    )
                self.use_apex = True

    def create_optimizer_and_scheduler(self, num_training_steps, learning_rate=None, warmup_steps=None, weight_decay=None,model_num=None):
        """
        Based on Transformers' default one, we add fixing layer option where the bottom n layers' parameters
        are fixed and only the top layers are further fine-tuned.
        """
        if learning_rate is None:
            learning_rate = self.args.learning_rate

        if warmup_steps is None:
            warmup_steps = self.args.warmup_steps

        if weight_decay is None:
            weight_decay = self.args.weight_decay

        if self.optim[model_num] is None:
            params = {}
            prompt_params = {}
            for n, p in self.model_list[model_num].named_parameters():

                if 'prompt_embeddings' in n:
                    prompt_params[n] = p
                    continue
                if self.args.fix_layers > 0:
                    if 'encoder.layer' in n:
                        try:
                            layer_num = int(n[n.find('encoder.layer') + 14:].split('.')[0])
                        except:
                            print(n)
                            raise Exception("")
                        if layer_num >= self.args.fix_layers:
                            print('yes', n)
                            params[n] = p
                        else:
                            print('no ', n)
                    elif 'embeddings' in n:
                        print('no ', n)
                    else:
                        print('yes', n)
                        params[n] = p
                else:
                    if p.requires_grad:
                        params[n] = p
            no_decay = ["bias", "LayerNorm.weight"]


            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in params.items() if not any(nd in n for nd in no_decay)],
                    "weight_decay": weight_decay,
                },
                {
                    "params": [p for n, p in params.items() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
                {
                    "params": [p for n, p in prompt_params.items() if any(nd in n for nd in no_decay)],
                    "lr": self.args.prompt_learning_rate,
                }
            ]

            self.optim[model_num] = AdamW(
                optimizer_grouped_parameters,
                lr=learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )
        self.count_params()
        if self.lr_scheduler_list[model_num] is None:
            self.lr_scheduler_list[model_num] = get_linear_schedule_with_warmup(
                self.optim[model_num], num_warmup_steps=warmup_steps, num_training_steps=num_training_steps
            )



    def count_params(self):
        total_param = 0
        update_parameter = []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            update_parameter.append(n)
            if 'decoder' in n:
                if self.model.data_args.task_name == 'mnli':
                    total_param += p.size(-1) * 3
                else:
                    total_param += p.size(-1) * 2
            else:
                total_param += p.numel()
        print("Model parameters number is {}".format(total_param))
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behaviort.
        """

        outputs = model(**inputs)
        # self.embedding_all.append(outputs[-1])
        # self.label.append(inputs['labels'])

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        loss = outputs[0]

        return (loss, outputs) if return_outputs else loss

    def compute_contrast_loss(self, model, inputs, loss_weight=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        if loss_weight is None:
            contra_loss = model.get_constrast_loss(**inputs)
        else:
            mask = loss_weight > 0
            inputs = {key: inputs[key][mask] for key in inputs}
            contra_loss = model.get_constrast_loss(**inputs)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        # if self.args.past_index >= 0:
        #     self._past = outputs[self.args.past_index]

        return contra_loss

    def compute_adv_loss(self, model, inputs, loss_weight=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        NEED TO FIX
        """

        adv_loss = model.get_adv_loss(**inputs)
        if loss_weight is not None:
            adv_loss = (adv_loss.mean(-1) * loss_weight).sum()
        else:
            adv_loss = adv_loss.mean()

        return adv_loss

    def compute_un_loss(self, model, un_inputs, loss_weight=None, soft_label=None, fwd_type=None, features=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        NEED TO FIX
        # """
        # if self.label_smoother is not None and "psuedo_labels" in un_inputs:
        #     labels = un_inputs.pop("psuedo_labels")
        # else:
        #     labels = None

        un_labels = un_inputs['labels']
        del un_inputs['labels']

        outputs = model(**un_inputs)
        un_logits = outputs[0]

        if soft_label is not None:
            soft_label_way = soft_label == 1
        else:
            soft_label_way = (self.args.soft_label == 1)

        if soft_label_way:
            loss = F.kl_div(F.log_softmax(un_logits, dim=-1, dtype=torch.float32),
                            un_labels, reduction='none').sum(-1)
        else:
            loss = F.cross_entropy(un_logits, un_labels, reduction='none')

        if fwd_type is not None:
            if fwd_type == 4:
                sim = nn.CosineSimilarity(dim=-1)
                feature_loss = sim(features, outputs[1])
                loss = loss + feature_loss


        if loss_weight is not None:
            loss = (loss * loss_weight).sum()
        else:
            loss = loss.mean()


        return loss,outputs[-1]

    def meta(self, model, un_inputs, sampling_step=1, epsilon=1e-6, learning_rate=None, use_soft_label=True):

        model.eval()
        opti_param = [p for p in model.parameters() if p.requires_grad]
        if learning_rate is None:
            inner_opt = torch.optim.SGD(opti_param, lr=self.args.learning_rate)
        else:
            inner_opt = torch.optim.SGD(opti_param, lr=learning_rate)
        un_labels = un_inputs['labels']

        if use_soft_label is  None:
            use_soft_label = self.args.soft_label
            #un_labels = un_inputs['labels']


        for i in range(sampling_step):
            self.clear_memory()
            try:
                meta_inputs = next(self.meta_train_iter)
            except:
                self.meta_train_iter = self.meta_train_dataloader.__iter__()
                meta_inputs = next(self.meta_train_iter)

            with higher.innerloop_ctx(
                    model, inner_opt, copy_initial_weights=True,
            ) as (fnet, diffopt):


                un_outputs = fnet(**un_inputs)
                un_logits = un_outputs[1]

                if use_soft_label == 1:
                    un_loss = F.kl_div(F.log_softmax(un_logits, dim=-1, dtype=torch.float32),
                                    un_labels, reduction='none').sum(-1)
                else:
                    un_loss = F.cross_entropy(un_logits, un_labels, reduction='none')
                weight = torch.zeros(un_loss.size(), requires_grad=True).to(self.accelerator.device)
                new_loss = (un_loss * weight).sum()

                diffopt.step(new_loss)

                meta_outputs = fnet(**meta_inputs)
                loss = meta_outputs["loss"] if isinstance(meta_outputs, dict) else meta_outputs[0]

                # graph = make_dot(loss, params=dict(fnet.named_parameters()))
                # # 可视化计算图并保存为PDF文件
                # graph.render("compute_graph")
                grad_eps = torch.autograd.grad(loss, weight, only_inputs=True)[0].detach()

            if i == 0:
                loss_weight = (- grad_eps)
            else:
                loss_weight += (-grad_eps)

            model.zero_grad()

        loss_weight = torch.clamp(loss_weight, min=0)

        #loss_weight = (loss_weight > 0) + 0
        norm_c = torch.sum(loss_weight) + epsilon
        if norm_c != 0:
            loss_weight = loss_weight / norm_c
        else:
            loss_weight = loss_weight
        model.train()
        return loss_weight.detach()

    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare :obj:`inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.args.device)

        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past

        return inputs

    #wxy
    def dist(pro, unlabeldata, mask):
    # S [class, embed_dim], unlabeldata [num_of_sent, num_of_tokens, embed_dim]  mask [num_of_sent, num_of_tokens]
        assert unlabeldata.size()[:2] == mask.size()
        mask = mask.view(-1) == 1
        print("maskshape", mask.shape)
        unlabeldata = unlabeldata.view(-1, unlabeldata.size(-1))[mask]
        print("unlabeldata", unlabeldata.shape)
        # [num_of_all_text_tokens, embed_dim]
        return (torch.pow(pro.unsqueeze(0) - unlabeldata.unsqueeze(1), 2)).sum(2)

    def pseudo_data_selection(self, model, un_inputs, un_meta,unembedding,learning_rate=None, use_soft_label=None):
        loss_weight = None
        proto=self.proto#wxy
        if proto!=None:
            dis=[]
            if protonet:
                dis = euclidean_dist(unembedding, proto)
            else:
                for i in range(len(unembedding)):
                    disi=[]
                    for j in range(len(proto)):
                        disi.append(float(torch.pow(unembedding[i]- proto[j], 2).sum(0)))
                        # disi.append(cos(unembedding[i],proto[j]))
                    dis.append(disi)
                dis=torch.tensor(dis)
            mask=torch.var(dis,dim=1).cuda()
        if self.args.psuedo_selection_opt == "meta" and self.delta > self.meta_st_warmup_steps:
            loss_weight = self.meta(model, un_inputs, sampling_step=self.args.sampling_steps, learning_rate=learning_rate, use_soft_label=use_soft_label)

        elif self.args.psuedo_selection_opt == "confidence" or (self.delta < self.meta_st_warmup_steps and  self.args.psuedo_selection_opt == "meta"):
            soft_labels = un_meta.get("soft_labels", None)
            # mask = torch.max(soft_labels, dim=-1)[0] > self.args.confidence_thresh#wxy
            loss_weight = mask / (mask.sum() + 1e-5)

        return loss_weight

    def assign_psuedo_label(self, un_inputs, model=None, soft_label=None, fwd_type=None,student_num=None):
        if model is None:
            teacher_model = self.teacher_model[student_num].to(self.accelerator.device)
            teacher_model.eval()
        else:
            teacher_model = model
            teacher_model.eval()


        with torch.no_grad():
            outputs = teacher_model(**un_inputs)
            unembeddding=outputs[-1]
            # Save past state if it exists
            # TODO: this needs to be fixed and made cleaner later.
            # NEED TO FIX

            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
            logits = logits.detach()

            if fwd_type is not None and fwd_type == 4:
                features = outputs[1].detach()

            soft_labels = F.softmax(logits, dim=-1)
            hard_labels = torch.max(F.softmax(logits, dim=-1), dim=-1)[1]

            if soft_label is not None:
                use_soft_label = (soft_label == 1)
            else:
                use_soft_label = (self.args.soft_label == 1)
            if use_soft_label == 1:
                if self.args.sharpen == 1:
                    sharpen_soft_labels = logits / self.args.temperature
                    un_inputs['labels'] = F.softmax(sharpen_soft_labels, dim=-1)
                else:
                    un_inputs['labels'] = soft_labels

            else:
                un_inputs['labels'] = hard_labels

        self.clear_memory()
        if fwd_type is not None and fwd_type == 4:
            return soft_labels, hard_labels, features
        # if protonet:
        #     unembeddding=teacher_model.lm_model.proto_head(outputs[-1])
        # return soft_labels, hard_labels,outputs[-1].detach()#wxy
        return soft_labels, hard_labels,unembeddding#wxy

    def get_un_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.
        Will use no sampler if :obj:`self.train_dataset` does not implement :obj:`__len__`, a random sampler
        (adapted to distributed training if necessary) otherwise.
        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.un_train_dataset is None:
            raise ValueError("Trainer: un_training requires a un_train_dataset.")
        un_train_sampler = self._get_un_train_sampler()

        return DataLoader(
            self.un_train_dataset,
            batch_size=self.args.un_train_batch_size,
            sampler=un_train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )

    def get_demo_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.
        Will use no sampler if :obj:`self.train_dataset` does not implement :obj:`__len__`, a random sampler
        (adapted to distributed training if necessary) otherwise.
        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_demo_dataset is None:
            raise ValueError("Trainer: un_training requires a un_train_dataset.")
        train_demo_sampler = self._get_train_demo_sampler()

        return DataLoader(
            self.train_demo_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_demo_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )

    def get_meta_train_dataloader(self, percentage=1) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.
        Will use no sampler if :obj:`self.train_dataset` does not implement :obj:`__len__`, a random sampler
        (adapted to distributed training if necessary) otherwise.
        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: un_training requires a un_train_dataset.")
        train_sampler = self._get_train_sampler()

        if percentage == 1:
            self.meta_train_dataset = self.train_dataset
        else:
            self.meta_dataset = copy.deepcopy(self.train_dataset)
            self.meta_train_dataset = copy.deepcopy(self.train_dataset)
            np.random.shuffle(self.meta_dataset.example_idx)
            self.meta_train_num = int(percentage * len(self.meta_dataset.example_idx))
            self.meta_train_dataset.example_idx = self.meta_dataset.example_idx[:self.meta_train_num]

        return DataLoader(
            self.meta_train_dataset,
            batch_size=self.args.meta_train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )

    def get_meta_valid_dataloader(self, percentage=1) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.
        Will use no sampler if :obj:`self.train_dataset` does not implement :obj:`__len__`, a random sampler
        (adapted to distributed training if necessary) otherwise.
        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: un_training requires a un_train_dataset.")
        train_sampler = self._get_train_sampler()

        if percentage == 1:
            self.meta_valid_dataset = self.train_dataset
        else:
            self.meta_valid_dataset = copy.deepcopy(self.train_dataset)
            self.meta_valid_dataset.example_idx = self.meta_dataset.example_idx[self.meta_train_num:]


        return DataLoader(
            self.meta_valid_dataset,
            batch_size=self.args.meta_train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )


    def _get_un_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        if not isinstance(self.un_train_dataset, collections.abc.Sized):
            return None
        elif is_torch_tpu_available():
            return get_tpu_sampler(self.un_train_dataset)
        else:
            return (
                RandomSampler(self.un_train_dataset)
                if self.args.local_rank == -1
                else DistributedSampler(self.un_train_dataset)
            )

    def _get_train_demo_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        if not isinstance(self.train_demo_dataset, collections.abc.Sized):
            return None
        elif is_torch_tpu_available():
            return get_tpu_sampler(self.train_demo_dataset)
        else:
            return (
                RandomSampler(self.train_demo_dataset)
                if self.args.local_rank == -1
                else DistributedSampler(self.train_demo_dataset)
            )
    def proto_sampler(self):
        return PrototypicalBatchSampler(self.train_dataset)
        
    def get_proto_dataloader(self, percentage=1) -> DataLoader:
        
        if self.train_dataset is None:
            raise ValueError("Trainer: un_training requires a un_train_dataset.")
        sampler = self.proto_sampler()

        self.proto_dataset = copy.deepcopy(self.train_dataset)
        # self.meta_train_dataset = copy.deepcopy(self.train_dataset)
        # np.random.shuffle(self.proto_dataset.example_idx)
        self.proto_train_num = len(self.proto_dataset.example_idx)
        # self.meta_train_dataset.example_idx = self.meta_dataset.example_idx[:self.meta_train_num]

        return DataLoader(
            self.proto_dataset,
            batch_size=20,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )

    def demon_condensation_step(self, model, demo_inputs=None, inputs=None):


        if inputs is not None:
            del inputs['labels']
        loss = 0
        un_meta = {}
        fwd_type = -1
        demo_inputs['fwd_type'] = fwd_type
        inputs['fwd_type'] = fwd_type
        features = None

        if fwd_type == 4:
            soft_labels, hard_labels, features = self.assign_psuedo_label(demo_inputs, model=model, soft_label=1, fwd_type=fwd_type)
            inputs['labels'] = soft_labels
        else:
            soft_labels, hard_labels = self.assign_psuedo_label(demo_inputs, model=model, soft_label=1)
            inputs['labels'] = soft_labels

        model.train()
        un_meta['soft_labels'] = soft_labels
        un_meta['hard_labels'] = hard_labels


        del inputs['fwd_type']

        loss_weight = self.pseudo_data_selection(model, inputs, un_meta, use_soft_label=1)

        inputs['fwd_type'] = fwd_type
        conden_loss = self.compute_un_loss(model, inputs, loss_weight=loss_weight, soft_label=1, fwd_type=fwd_type, features=features)
        loss = loss + conden_loss

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if isinstance(loss, int):
            return 0
        self.accelerator.backward(loss)
        return loss.detach().item()



    def training_step(self, model, inputs=None, un_inputs=None,student_num=None) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.
        Subclass and override to inject custom behavior.
        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """

        un_meta = {}
        model.train()

        if un_inputs is not None:
            if "labels" in un_inputs:
                del un_inputs['labels']

        loss = 0

        if inputs is not None:
            if self.loss_alpha != 0:
                if self.args.fp16 and _use_native_amp:
                    with autocast():
                        loss = self.compute_loss(model, inputs)
                else:
                    loss = self.compute_loss(model, inputs)

            if self.args.adv_opt == 1 or self.args.adv_opt == 3:
                adv_loss = self.compute_adv_loss(model, inputs)
                loss = loss + adv_loss

            if self.args.contrast_training == 1:
                contrast_loss = self.compute_contrast_loss(model, inputs)
                # print(contrast_loss)
                loss = loss + contrast_loss
            # logger.info("loss:{}".format(loss))
            if protonet:
                loss_pro=0
                # for step,proinputs in enumerate(self.pro_dataloader):
                if 0:
                    output=model(**self.proinputs)
                    embedding_all=output[-1]
                else:
                    model.lm_model.proto_head.train()
                    embedding_all=model.get_proto_embedding(**self.proinputs)
                label=self.proinputs['labels']
                loss_pro,acc=loss_fn(embedding_all.to(self.accelerator.device), target=label.to(self.accelerator.device),n_support=2,device=self.accelerator.device)
                # loss=loss_pro/10+loss
                # logger.info("proto loss:{}".format(loss_pro))
                self.accelerator.backward(loss_pro,retain_graph=True)
                self.optim[student_num].step()
                self.model.zero_grad()
                # model.lm_model.proto_head.zero_grad()
                # with amp.scale_loss(loss_pro, self.optimizer) as scaled_loss:
                #     scaled_loss.backward()
                
                

        if self.args.is_semi == 1 and (un_inputs is not None) and (len(un_inputs) != 0):
            if self.args.use_psuedo_label > 0:

                soft_labels, hard_labels,un_embedding = self.assign_psuedo_label(un_inputs,student_num=student_num)#wxy
            
                un_meta['soft_labels'] = soft_labels
                un_meta['hard_labels'] = hard_labels
                if self.args.soft_label == 1:
                    un_meta['pseudo_label'] = soft_labels
                else:
                    un_meta['pseudo_label'] = hard_labels

                loss_weight = self.pseudo_data_selection(model, un_inputs, un_meta,unembedding=un_embedding)#wxy
        
                un_loss,unembedding = self.compute_un_loss(model, un_inputs, loss_weight=loss_weight)
                loss = loss + un_loss
                self.consistent_embedding[student_num]=un_embedding
                
                mixup=self.args.mix
                if mixup:
                    mixloss=[]
                    mix_loss=0
                    un_embedding=model.lm_model.proto_head(unembedding)        
                    for i in range(len(hard_labels)):
                        if max(soft_labels[i])>0.5:
                            mixembedding=(un_embedding[i]+self.proto[hard_labels[i]])/2
                            
                            if 1:
                                #加入protohead后的版本，embedding是原型输出。mix是原型向量和mask过了protohead的值进行mix
                                dists = euclidean_dist(mixembedding.unsqueeze(0), self.proto)
                                log_p_y = F.log_softmax(-dists, dim=1).view(len(self.proto), -1).squeeze()
                                loss_1=-log_p_y[hard_labels[i]]
                            if 0:
                                #这一部分是原来没有加protohead的版本，embedding是robert的mask输出，过一个head映射到词表，然后根据类别在词表的id找到每个类别的当前输出，得到一个（len,class）的向量logits,在和label做一个交叉熵函数
                                prediction_mask_scores=self.model.lm_model.lm_head(mixembedding)
                                logits = []
                                for label_id in range(len(self.model.lm_model.label_word_list)):
                                    logits.append(prediction_mask_scores[self.model.lm_model.label_word_list[label_id]].unsqueeze(-1))
                                loss_fct = nn.CrossEntropyLoss()
                                logits = torch.cat(logits, -1)
                                loss_1=loss_fct(logits.view(-1, logits.size(-1)), hard_labels[i].view(-1,))*max(soft_labels[i])
                            mixloss.append(loss_1)
                            mix_loss=mix_loss+loss_1
                    if mix_loss!=0:
                        mix_loss=mix_loss/len(mixloss)
                        logger.info("mixloss: {}".format(mix_loss))
                        self.accelerator.backward(mix_loss,retain_graph=True)
                        self.summaryWriter.add_scalar("mixloss_linear",mix_loss,self.global_step)
                # if mixup: 
                #     # self.accelerator.backward(mixup.item(),retain_graph=True)
                #     loss=loss+mix_loss
                
            
            if self.args.adv_opt > 1:
                un_adv_loss = self.compute_adv_loss(model, un_inputs, loss_weight=loss_weight)
                loss = loss + un_adv_loss
            #无标签数据不用对比wxy
            if self.args.contrast_training > 2:
                un_inputs['labels'] = un_meta['hard_labels']
                contrast_loss = self.compute_contrast_loss(model, un_inputs, loss_weight=loss_weight)
                # print(contrast_loss)
                loss = loss + contrast_loss

        if self.args.n_gpu > 4:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if isinstance(loss, int):
            return 0
        self.accelerator.backward(loss,retain_graph=True)

        return loss.detach().item()
    
    def proto_loss(self,model):
        epoch_iterator = self.train_dataloader
        embedding_all=[]
        label=[]#silver
        for step, inputs in enumerate(epoch_iterator):
            output=model(**inputs)
            if step==0:
                embedding_all=output[-1]
                label=inputs['labels']
            else:
                embedding_all=torch.cat((embedding_all,output[-1]))
                label=torch.cat((label,inputs['labels']))
            # embedding_all.append(output[-1].cpu())
            # label.append(inputs['labels'].cpu())
        embedding_input=torch.stack(embedding_all,dim=1).view(len(embedding_all)*embedding_all[0].shape[0],-1)
        target=torch.stack(label).view(len(label)*label[0].shape[0],-1)
        loss_pro,acc=loss_fn(embedding_input.to(self.accelerator.device), target=target.to(self.accelerator.device),
                        n_support=8,device=self.accelerator.device)
        return loss_pro
    
    def update_teacher(self, model,student_num):

        model_file = os.path.join(self.output_list[student_num], "pytorch_model.bin")
        if self.teacher_model[student_num] is None:
            self.teacher_model[student_num] = copy.deepcopy(model)

        if self.args.mean_teacher:
            pass
        else:
            if model_file is not None and os.path.exists(model_file):
                logger.info('loading model from {}'.format(model_file))
                self.teacher_model[student_num].load_state_dict(torch.load(model_file))


    def re_init(self,student_num):
        config = AutoConfig.from_pretrained(
            self.teacher_model[student_num].model_args.config_name if self.teacher_model[student_num].model_args.config_name else self.teacher_model[student_num].model_args.model_name_or_path,
            num_labels=self.teacher_model[student_num].num_labels,
            finetuning_task=self.teacher_model[student_num].data_args.task_name,
            cache_dir=self.teacher_model[student_num].model_args.cache_dir,
        )

        if 'prompt' in self.teacher_model[student_num].model_args.few_shot_type:
            model_fn = LMForPromptFinetuning

        elif self.teacher_model[student_num].model_args.few_shot_type == 'finetune':
            model_fn = AutoModelForSequenceClassification
        else:
            raise NotImplementedError


        if self.teacher_model[student_num].data_args.prompt:
            self.model = model_fn(config, self.teacher_model[student_num].model_args, self.teacher_model[student_num].data_args)
        else:
            self.model = model_fn.from_pretrained(
                self.teacher_model[student_num].model_args.model_name_or_path,
                from_tf=bool(".ckpt" in self.teacher_model[student_num].model_args.model_name_or_path),
                config=config,
                cache_dir=self.teacher_model[student_num].model_args.cache_dir,
            )
            self.model.model_args = self.teacher_model[student_num].model_args
            self.model.data_args = self.teacher_model[student_num].data_args
        lora=1
        if lora:
            peft_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1)
            self.model=get_peft_model(self.model, peft_config)
        self.wipe_memory(student_num)



    def clear_memory(self):
        gc.collect()
        torch.cuda.empty_cache()

    def wipe_memory(self,student_num):  # DOES WORK
        self._optimizer_to(torch.device('cpu'),student_num)
        # del self.optimizer
        del self.optim[student_num]
        self.clear_memory()
        # self.optimizer=None
        self.optim[student_num]=None

    def _optimizer_to(self, device,student_num):
        for param in self.optim[student_num].state_dict()['state'].values():
            # Not sure there are any global tensors in the state dict
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(device)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(device)

    def protodata_test(self):
        proto=[]#wxy
        pro_dataloader = self.get_proto_dataloader()
        epoch_iterator = pro_dataloader
        # if self.global_step >=self_training_start_iter and epoch%10==0:
        for step,input in enumerate(epoch_iterator):
            pass

    
    def prototype(self,data):
        proto=[]#wxy
        epoch_iterator = data
        embedding_=np.array([])
        label_=np.array([])#silver
        # if self.global_step >=self_training_start_iter and epoch%10==0:
        proto_model = copy.deepcopy(self.model)
        model_file = os.path.join(self.args.output_dir, "pytorch_model.bin")
        if os.path.exists(model_file):
            proto_model.load_state_dict(torch.load(model_file))
        proto_model = proto_model.to(self.accelerator.device)
    
        proto_model.eval()
        for step, inputs in enumerate(epoch_iterator):
            if protonet:
                # proto_model.lm_model.proto_head.eval()
                mask_embedding=proto_model.get_proto_embedding(**inputs).detach()
            else:
                output=proto_model(**inputs)
                mask_embedding=output[-1].detach()
            tru_label=inputs['labels'].cpu()
            if step==0:
                embedding_=mask_embedding
                label_=tru_label.cpu()
            else:
                embedding_=torch.cat((embedding_,mask_embedding))
                label_=torch.cat((label_,tru_label))
        for label in range(torch.max(label_)+1):
            proto.append(torch.mean(embedding_[label_==label], 0))
        proto = torch.stack(proto)

        return proto



    def get_train_ds_config(offload=0,
                            stage=2,
                            enable_hybrid_engine=False,
                            inference_tp_size=1,
                            release_inference_cache=False,
                            pin_parameters=True,
                            tp_gather_partition_size=8):

        device = "cpu" if offload else "none"
        zero_opt_dict = {
            "stage": stage,
            "offload_param": {
                "device": device
            },
            "offload_optimizer": {
                "device": device
            },
            "stage3_param_persistence_threshold": 1e4,
            "stage3_max_live_parameters": 3e7,
            "stage3_prefetch_bucket_size": 3e7,
            "memory_efficient_linear": False
        }
        return {
            "train_batch_size": 2,
            "train_micro_batch_size_per_gpu": 2,
            "steps_per_print": 10,
            "zero_optimization": 2,
            "fp16": {
                "enabled": True,
                "loss_scale_window": 100
            },
            "gradient_clipping": 1.0,
            "prescale_gradients": False,
            "wall_clock_breakdown": False,
            "hybrid_engine": {
                "enabled": enable_hybrid_engine,
                "inference_tp_size": inference_tp_size,
                "release_inference_cache": release_inference_cache,
                "pin_parameters": pin_parameters,
                "tp_gather_partition_size": tp_gather_partition_size,
            }
        }

    def train(self, model_path=None, dev_objective=None):
        """
        Main training entry point.

        The training logic is directly borrowed from transformers.Trainer (version 3.0.2).
        Add early stopping.
        """


        self._memory_tracker.start()

        self.teacher_model = [None,None]
        self.consistent_embedding=[None,None]
        self.best_dir = None
        args = self.args
        self.is_in_train = True
        self.objective = [-float("inf"),-float("inf")]
        start_time = time.time()
        self.state.max_steps = self.args.max_steps
        self.dev_objective = dev_objective if dev_objective is not None else default_dev_objective
        self.teacherobjective=[-float("inf"),-float("inf")]
        global protonet
        protonet=args.protonet
        # Data loading.
        train_dataloader = self.get_train_dataloader()

        num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps

        self_training_start_iter = self.args.self_training_start_epoch * int(
            len(train_dataloader) // self.args.gradient_accumulation_steps)

        finetune_teacher_steps = self.args.finetune_teacher_epoch * int(
            len(train_dataloader) // self.args.gradient_accumulation_steps)

        update_teacher_steps = (self.args.update_teacher_steps // self.args.un_gradient_accumulation_steps)

        if self.args.is_semi == 1:
            self_training_total_steps = update_teacher_steps
            if self.args.semi_finetune:
                self_training_total_steps = self_training_total_steps  + finetune_teacher_steps
            self.un_train_dataloader = self.get_un_train_dataloader()
        else:
            self.un_train_dataloader = None
        if num_update_steps_per_epoch == 0:
            num_update_steps_per_epoch = 1
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
                self.args.max_steps % num_update_steps_per_epoch > 0
            )
        elif self.args.is_semi == 1 and self.args.self_training_session > 0:
            t_total =  self_training_total_steps * self.args.self_training_session + self_training_start_iter
            self.args.max_steps = t_total
            num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
                self.args.max_steps % num_update_steps_per_epoch > 0
            )
        else:
            t_total = int(len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs

        self.model_list.append(self.model)
        self.model_list.append(self.another_model)
        self.optim=[None,None]
        self.lr_scheduler_list=[None,None]
        self.meta_st_warmup_steps = int(self.args.meta_st_warmup * self.args.update_teacher_steps)
        lora=1
        for i in range(2):
            if lora:
                self.model_list[i].freeze_lora_component()
            if self.args.update_k_layers != -1:
                
                self.model_list[i].freeeze_lm_k_layers(self.args.update_k_layers)

            if self.args.update_component != "none":
                self.model_list[i].freeze_lm_component(self.args.update_component)


            if self.args.is_semi == 1:
                self.create_optimizer_and_scheduler(num_training_steps=self_training_start_iter,model_num=i)
            else:
                self.create_optimizer_and_scheduler(num_training_steps=t_total)
        self.t_total = t_total
        self.meta_train_dataloader = self.get_meta_train_dataloader()
        self.meta_valid_dataloader = self.get_meta_valid_dataloader()
        


        un_inputs = None


        ## new add
        self.train_dataloader = train_dataloader
        train_dataloader = self.get_train_dataloader()
        self.demo_train_dataloader = self.get_demo_train_dataloader()
        
        if 1:
            self.train_dataset=self.another_train_dataset
            self.another_train_dataloader=self.get_train_dataloader()
            self.model_list[0], self.optim[0], self.model_list[1], self.optim[1],self.train_dataloader, self.another_train_dataloader,self.un_train_dataloader\
                = self.accelerator.prepare(
                self.model_list[0], self.optim[0], self.model_list[1], self.optim[1],self.train_dataloader, self.another_train_dataloader,self.un_train_dataloader
            )
        else:
            self.model, self.optimizer, self.train_dataloader,  self.meta_train_dataloader, self.meta_valid_dataloader, self.un_train_dataloader, self.demo_train_dataloader\
                = self.accelerator.prepare(
                self.model, self.optimizer, self.train_dataloader,  self.meta_train_dataloader, self.meta_valid_dataloader, self.un_train_dataloader, self.demo_train_dataloader
            )
            

        un_train_dataloader = self.un_train_dataloader
        train_dataloader = self.train_dataloader
        self.pro_dataloader = self.get_proto_dataloader()
        self.pro_dataloader=self.accelerator.prepare(self.pro_dataloader)
        self.meta_train_iter = self.meta_train_dataloader.__iter__()
        another_train_dataloader=self.another_train_dataloader


        # Check if saved optimizer or scheduler states exist
        if (
                model_path is not None
                and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
                and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            self.optimizer.load_state_dict(
                torch.load(os.path.join(model_path, "optimizer.pt"), map_location=self.args.device)
            )
            self.lr_scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))


        # Train
        if transformers.is_torch_tpu_available():
            total_train_batch_size = self.args.train_batch_size * xm.xrt_world_size()
        else:
            total_train_batch_size = (
                    self.args.train_batch_size
                    * self.args.gradient_accumulation_steps
                    * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
            )
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", self.num_examples(train_dataloader))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Unlabeled Gradient Accumulation steps = %d", self.args.un_gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        self.global_step = 0
        self.epoch = 0
        self.loss_alpha = 1
        epochs_trained = 0
        finetune = True
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if model_path is not None:
            # set global_step to global_step of last saved checkpoint from model path
            try:
                self.global_step = int(model_path.split("-")[-1].split("/")[0])
                epochs_trained = self.global_step // (len(train_dataloader) // self.args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = self.global_step % (
                        len(train_dataloader) // self.args.gradient_accumulation_steps
                )

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", self.global_step)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                self.global_step = 0#wxy
                logger.info("  Starting fine-tuning.")
        # deepspeed.initialize(args=args,  model=self.model,model_parameters=params,training_data=dataset)
        deep=0
        if deep:
            config_file = "/root/grad/831list/src/deepspeed_config.json"
            ds_config = self.get_train_ds_config()
            self.model, self.optimizer, _, _ = deepspeed.initialize(model=self.model,
            optimizer=self.optimizer,
            config_params=ds_config)
        if wxytest:
            num_train_epochs=50
        tr_loss = [torch.tensor(0.0).to(self.args.device),torch.tensor(0.0).to(self.args.device)]
        logging_loss_scalar = 0.0
        for i in range(2):
            self.model_list[i].zero_grad()
            self.dataloader=[]
            self.dataloader.append(self.train_dataloader)
            self.dataloader.append(self.another_train_dataloader)
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                # We just need to begin an iteration to create the randomization of the sampler.
                for _ in train_dataloader:
                    break
        train_iterator = trange(
            epochs_trained, int(num_train_epochs), desc="Epoch"
        )
        demo_inputs=None
        session_num = [0,0]
        proto=[]
        self.summaryWriter = SummaryWriter("loss/log_proto")   
        self.proto_list=[None,None]
        self.output_list=[None,None]
        self.output_list[0]=self.args.output_dir
        self.output_list[1]=self.args.output_dir2
                    
    
                
                
                
        for epoch in train_iterator:
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader
            
            
            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)
            un_train_iter = None
            
            # if protonet:
            #     self.embedding_all=[]
            #     self.label=[]
            
            for step, inputs in enumerate(epoch_iterator):
                if protonet:
                    self.epochtrained=epoch
                    try:
                        self.proinputs = next(pro_train_iter)
                    except:
                        pro_train_iter = self.pro_dataloader.__iter__()
                        self.proinputs = next(pro_train_iter)
                delta = (self.global_step - self_training_start_iter)
                
                for student_num in range(2):
                    # self.model=self.model_list[student_num]
                    # self.optimizer=self.optim[student_num]
                    # self.lr_scheduler=self.lr_scheduler_list[student_num]
                    if self.args.use_last_epoch and self.global_step >self_training_start_iter and  self.global_step%200==0:
                        proto=self.prototype(self.dataloader[student_num])
                        self.proto_list[student_num]=proto
                    if self.args.is_semi == 1 and (delta >= 0):
                        # if delta == 0:
                        #     self.objective = -float("inf")
                        delta = delta % self_training_total_steps
                        self.delta = delta

                        if  self.args.semi_finetune:
                            if delta >= update_teacher_steps:
                                if delta == update_teacher_steps:

                                    logger.info("######### Start finetuning #########")

                                    if not self.args.use_last_epoch:
                                        model_file = os.path.join(self.output_list[student_num], "pytorch_model.bin")
                                        if model_file is not None and os.path.exists(model_file):
                                            logger.info('loading model from {}'.format(model_file))
                                            self.model_list[student_num].load_state_dict(torch.load(model_file))


                                finetune = True

                                self.loss_alpha = 1
                            else:
                                if delta == 0 and (self.args.semi_finetune or self.args.psuedo_selection_opt == 'meta'):
                                    if self.args.psuedo_selection_opt == 'meta':
                                        logger.info("######### Start meta re-weighting #########")

                                    if self.args.use_last_epoch:
                                        self.save_model(self.output_list[student_num])
                                    session_num[student_num] += 1
                                    if session_num[student_num] > self.args.self_training_session:
                                        break

                                    #wxy
                                    # proto_iterator = train_dataloader
                                    proto=self.prototype(self.dataloader[student_num])
                                    self.proto_list[student_num]=proto
                                    
                                finetune = False
                                self.loss_alpha = 0

                        if delta == 0:
                            self.model_list[student_num].zero_grad()
                            if self.args.use_psuedo_label > 0:
                                if 1 and self.objective[student_num]>=self.teacherobjective[student_num]:
                                    self.teacherobjective[student_num]=self.objective[student_num]
                                    self.update_teacher(self.model_list[student_num],student_num)

                            if self.args.re_init or self.args.psuedo_selection_opt == 'meta':
                                #这里需要修改，重新加入lora之后冻结了参数
                                logger.info('##### RE INIT MODEL #########')
                            
                                self.objective[student_num] = -float("inf")
                                self.re_init(student_num)
                                if lora:
                                    self.model_list[student_num].freeze_lora_component()
                                if self.args.freeze_encoder:
                                    logger.info("Freeze language model encoder")
                                    self.model_list[student_num].freeze_lm_encoder()

                                elif self.args.only_train_bias:
                                    logger.info("only finetune bias")
                                    self.model_list[student_num].freeze_lm_finetune_bias()

                                elif self.args.update_component != "none":
                                    self.model_list[student_num].freeze_lm_component(self.args.update_component)

                                semi_learning_rate = self.args.semi_learning_rate
                                semi_warmup_steps = self.args.semi_warmup_ratio * update_teacher_steps
                                semi_weight_decay = self.args.semi_weight_decay
                                # self.args.learning_rate = self.args.prompt_learning_rate
                                if 1:
                                    self.optim[student_num] = None
                                    self.lr_scheduler_list[student_num] = None
                                else:
                                    self.optimizer = None
                                    self.lr_scheduler = None
                                t_total = self_training_total_steps
                                self.create_optimizer_and_scheduler(num_training_steps=t_total, learning_rate=semi_learning_rate, warmup_steps=semi_warmup_steps, weight_decay=semi_weight_decay,model_num=student_num)
                                self.t_total = t_total
                                self.train_dataloader = self.accelerator.prepare(self.get_train_dataloader())
                                epoch_iterator = self.train_dataloader
                                self.model_list[student_num] = self.model_list[student_num].to(self.accelerator.device)

                        if un_train_iter is None:
                            un_train_iter = un_train_dataloader.__iter__()
                        if student_num == 0:
                            try:
                                un_inputs = next(un_train_iter)
                            except:
                                un_train_iter = un_train_dataloader.__iter__()
                                un_inputs = next(un_train_iter)

                    # Skip past any already trained steps if resuming training
                    if steps_trained_in_current_epoch > 0:
                        steps_trained_in_current_epoch -= 1
                        continue

                    if un_inputs is not None and (not finetune):
                        # try:
                        tr_loss[student_num] += self.training_step(self.model_list[student_num],  un_inputs=un_inputs,student_num=student_num)
                        # except:
                        #     print("One error here")
                            # self.clear_memory()
                        tr_loss[student_num] += self.training_step(self.model_list[student_num], inputs=inputs,student_num=student_num)
                    elif demo_inputs is not None and self.args.demo_condon == 1:
                        tr_loss += self.demon_condensation_step(self.model, demo_inputs=demo_inputs, inputs=inputs)
                    else:
                        tr_loss[student_num] += self.training_step(self.model_list[student_num], inputs,student_num=student_num)
                    self.current_flos += float(self.floating_point_ops(inputs))

                    if (finetune and (step + 1) % self.args.gradient_accumulation_steps == 0) or (not finetune and (step + 1) % self.args.un_gradient_accumulation_steps == 0) or (
                            # last step in epoch but step is always smaller than gradient_accumulation_steps
                            len(epoch_iterator) <= self.args.gradient_accumulation_steps
                            and (step + 1) == len(epoch_iterator)
                    ):

                        norm = torch.nn.utils.clip_grad_norm_(self.model_list[student_num].parameters(), self.args.max_grad_norm)
                        self.optim[student_num].step()
                        self.lr_scheduler_list[student_num].step()
                        self.model_list[student_num].zero_grad()
                        # self.model_list[student_num]=self.model
                        if student_num==1:
                            self.global_step += 1
                        self.epoch = epoch + (step + 1) / len(epoch_iterator)

                        if (self.args.logging_steps > 0 and self.global_step % self.args.logging_steps == 0) or (
                                self.global_step == 1 and self.args.logging_first_step
                        ):
                            logs = {}
                            tr_loss_scalar = tr_loss[student_num].item()
                            logs["loss"] = (tr_loss_scalar - logging_loss_scalar) / self.args.logging_steps
                            logs["norm"] = norm.item()
                            # backward compatibility for pytorch schedulers
                            logs["learning_rate"] = (
                                self.lr_scheduler_list[student_num].get_last_lr()[0]
                                if version.parse(torch.__version__) >= version.parse("1.4")
                                else self.lr_scheduler_list[student_num].get_lr()[0]
                            )
                            logging_loss_scalar = tr_loss_scalar

                            self.log(logs)

                        # ----------------------------------------------------------------------
                        # BEGIN CHANGES.
                        # ----------------------------------------------------------------------

                        if self.args.use_last_epoch: # and not (self.args.semi_finetune and self.args.is_semi == 1):
                            continue
                        if self.global_step % self.args.eval_steps == 0 and self.global_step>0:
                            output = self.evaluate()
                            metrics = output.metrics
                            objective = self.dev_objective(metrics)
                            logger.info("wxy student {} Dev result:  {}".format(student_num,objective))
                            if self.global_step % 1000 == 0 and self.global_step>0:
                                test=self.evaluate(eval_dataset=self.test_dataset)
                                testobj=self.dev_objective(test.metrics)
                                logger.info("wxy student {} test result: {}".format(student_num,testobj))
                            if objective >= self.objective[student_num]:
                                logger.info("wxy student {} self.teacher objective result: {}".format(student_num,self.teacherobjective[student_num]))
                                logger.info("Best student {} dev result: {}".format(student_num,objective))
                                self.objective[student_num] = objective
                                self.save_model(self.output_list[student_num])
                                
                            
                            #wxy
                            
                            proto=self.prototype(self.dataloader[student_num])
                            self.proto_list[student_num]=proto
                            self.proto=proto

                                # ----------------------------------------------------------------------
                        # END CHANGES.
                        # ----------------------------------------------------------------------


                    if self.args.max_steps > 0 and self.global_step > self.args.max_steps or \
                        (session_num[student_num] > self.args.self_training_session and self.args.is_semi == 1):
                        break
                train_loss=[0,0]
                if self.args.is_semi == 1 and (delta >= 0):
                    #del un_inputs['labels']

                    self.consistent_embedding[0] = self.model_list[0](**un_inputs)[-1]
                    self.consistent_embedding[1] = self.model_list[1](**un_inputs)[-1]
                    for student_num in range(2):
                        mutual_loss = 0.0
                        loss_mse = nn.MSELoss()
                        for other_student in range(2):
                            if other_student != student_num:
                                mutual_loss += loss_mse(self.consistent_embedding[student_num],self.consistent_embedding[other_student].detach())
                        # mutual_loss.backward()
                        self.accelerator.backward(mutual_loss,retain_graph=True)
                        
                        self.optim[student_num].step()
                        self.lr_scheduler_list[student_num].step()
                        self.model_list[student_num].zero_grad()

                        train_loss[student_num] += mutual_loss.item() * args.gradient_accumulation_steps
                    self.clear_memory()
            
            
            if self.args.use_last_epoch:  # and not (self.args.semi_finetune and self.args.is_semi == 1):
                continue

            if self.args.is_semi == 1:
                continue

            output = self.evaluate()
            metrics = output.metrics

            objective = self.dev_objective(metrics)
            logger.info("Dev result: {}".format(objective))

            if objective > self.objective:
                logger.info("self.objective result: {}".format(self.objective))
                logger.info("Best dev result: {}".format(objective))
                self.objective = objective
                self.save_model(self.args.output_dir)

            if self.args.max_steps > 0 and self.global_step > self.args.max_steps or \
                    (session_num[student_num] > self.args.self_training_session and self.args.is_semi == 1):
                train_iterator.close()
                break
            if self.args.tpu_metrics_debug or self.args.debug:
                # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                xm.master_print(met.metrics_report())

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")
        #wxy
        if self.args.use_last_epoch:
            self.save_model(self.args.output_dir)

        metrics = speed_metrics("train", start_time, self.state.max_steps)
        self.store_flos()
        self.log(metrics)
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        # add remaining tr_loss
        self._total_loss_scalar += tr_loss[0].item()

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        return TrainOutput(self.global_step, tr_loss / self.state.global_step, metrics)
        # return TrainOutput(self.global_step, tr_loss / self.global_step), self.objective

    """
    Difference compared to original implementation: return output instead of output.metrics (so there is also the logits)
    """

    def evaluate(self, eval_dataset: Optional[Dataset] = None) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are
        task-dependent (pass it to the init :obj:`compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed. It must implement
                the :obj:`__len__` method.

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions.
        """
        if eval_dataset is not None and not isinstance(eval_dataset, collections.abc.Sized):
            raise ValueError("eval_dataset must implement __len__")

        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        output = self.prediction_loop(eval_dataloader, description="Evaluation")

        self.log(output.metrics)

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        return output
