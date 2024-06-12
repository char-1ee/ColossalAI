from transformers.models.bloom.modeling_bloom import BloomBlock, BloomForCausalLM, BloomModel

from colossalai.inference.config import RPC_PARAM
from colossalai.inference.modeling.models.nopadding_bloom import (
    NopadBloomAttention,
    NopadBloomMLP,
    bloom_block_forward,
    bloom_causal_lm_forward,
    bloom_model_forward,
)
from colossalai.shardformer.policies.base_policy import ModulePolicyDescription, SubModuleReplacementDescription
from colossalai.shardformer.policies.bloom import BloomForCausalLMPolicy


class NoPaddingBloomModelInferPolicy(BloomForCausalLMPolicy, RPC_PARAM):
    def __init__(self) -> None:
        super().__init__()

    def module_policy(self):
        policy = super().module_policy()

        if self.shard_config.enable_tensor_parallelism:
            # TODO: tp
            pass
        else:
            decoder_attribute_replacement = None

        policy[BloomBlock] = ModulePolicyDescription(
            attribute_replacement=decoder_attribute_replacement,
            sub_module_replacement=[
                SubModuleReplacementDescription(
                    suffix="mlp",
                    target_module=NopadBloomMLP,
                ),
                SubModuleReplacementDescription(
                    suffix="self_attention",
                    target_module=NopadBloomAttention,
                    kwargs={
                        "model_shard_infer_config": self.shard_config.extra_kwargs["model_shard_infer_config"],
                    },
                ),
            ],
        )

        # policy[BloomForCausalLM] = ModulePolicyDescription(
        #     sub_module_replacement=[
        #         SubModuleReplacementDescription(
        #             suffix="lm_head",

        #         )
        #     ],
        # )

        self.append_or_create_method_replacement(
            description={"forward": bloom_causal_lm_forward},
            policy=policy,
            target_key=BloomForCausalLM,
        )
        self.append_or_create_method_replacement(
            description={"forward": bloom_model_forward},
            policy=policy,
            target_key=BloomModel,
        )
        self.append_or_create_method_replacement(
            description={"forward": bloom_block_forward},
            policy=policy,
            target_key=BloomBlock,
        )

        return policy

    def postprocess(self):
        return self.model

    def to_rpc_param(self) -> str:
        return __class__.__name__

    @staticmethod
    def from_rpc_param() -> "NoPaddingBloomModelInferPolicy":
        return NoPaddingBloomModelInferPolicy()
