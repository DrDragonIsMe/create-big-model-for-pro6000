#!/usr/bin/env python3
"""WanVideoWrapper multitalk/wav2vec2.py 兼容性补丁：
transformers 5.x 的 Wav2Vec2Encoder 不再返回 hidden_states，
用 forward hook 手动收集各层输出。幂等，可重复执行。"""
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else \
    "/opt/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper/multitalk/wav2vec2.py"

OLD = """        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]"""

NEW = """        # [patch] transformers 5.x 兼容：encoder 不再返回 hidden_states，用 hook 收集
        _collected = []
        _hooks = []
        if output_hidden_states and hasattr(self.encoder, "layers"):
            def _hs_hook(mod, args, output):
                _collected.append(output[0] if isinstance(output, tuple) else output)
            for _layer in self.encoder.layers:
                _hooks.append(_layer.register_forward_hook(_hs_hook))
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        for _h in _hooks:
            _h.remove()

        hidden_states = encoder_outputs[0]
        if output_hidden_states and _collected:
            encoder_outputs.hidden_states = (hidden_states,) + tuple(_collected)"""

src = open(PATH).read()
if "[patch] transformers 5.x" in src:
    print("已打过补丁")
elif OLD in src:
    open(PATH, "w").write(src.replace(OLD, NEW))
    print("补丁已应用:", PATH)
else:
    sys.exit("未找到目标代码块，wrapper 版本可能已变化，请人工检查")


