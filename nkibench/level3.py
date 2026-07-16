"""
NKIGen-Bench Level 3 tasks -- full model components (50 tasks).

Each task represents a complete neural network building block such as a
Transformer encoder layer, GPT-2 block, ResNet bottleneck, or Mamba block.
These tasks require generating NKI kernels that span many operations and
demand sophisticated tiling, pipelining, and engine orchestration.

Categories:
  mlp_components, transformer_components, cnn_components, ssm_components,
  vision_components, specialized_components
"""

from __future__ import annotations

from typing import List

from . import NKIBenchTask

_COUNTER = 0


def _make_task(
    name: str,
    category: str,
    model_code: str,
    inputs_code: str,
    nki_metadata: dict | None = None,
    expected_speedup: tuple[float, float] = (1.3, 8.0),
) -> NKIBenchTask:
    global _COUNTER
    _COUNTER += 1
    task_id = f"L3_{category}_{_COUNTER:03d}"
    return NKIBenchTask(
        task_id=task_id,
        name=name,
        level=3,
        category=category,
        model_class_code=model_code,
        get_inputs_code=inputs_code,
        nki_metadata=nki_metadata or {},
        expected_speedup_range=expected_speedup,
    )


# ============================================================================
# MLP components (6 tasks)
# ============================================================================

def _mlp_component_tasks() -> List[NKIBenchTask]:
    tasks: List[NKIBenchTask] = []

    # 2-layer MLP with dropout and layer_norm
    for suffix, D, FF, drop in [
        ("mlp_2layer_768", 768, 3072, 0.1),
        ("mlp_2layer_1024", 1024, 4096, 0.1),
    ]:
        model_code = (
            "import torch\n"
            "import torch.nn as nn\n"
            "import torch.nn.functional as F\n\n"
            "class Model(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            f"        self.ln = nn.LayerNorm({D})\n"
            f"        self.fc1 = nn.Linear({D}, {FF})\n"
            f"        self.fc2 = nn.Linear({FF}, {D})\n"
            f"        self.dropout = nn.Dropout({drop})\n\n"
            "    def forward(self, x):\n"
            "        residual = x\n"
            "        x = self.ln(x)\n"
            "        x = self.dropout(F.gelu(self.fc1(x)))\n"
            "        x = self.dropout(self.fc2(x))\n"
            "        return x + residual\n"
        )
        inputs_code = (
            "import torch\n\n"
            "def get_inputs():\n"
            f"    x = torch.randn(8, 128, {D}, dtype=torch.float32)\n"
            "    return [x]\n"
        )
        tasks.append(_make_task(
            name=suffix,
            category="mlp_components",
            model_code=model_code,
            inputs_code=inputs_code,
            nki_metadata={
                "engine_hints": ["tensor_engine", "vector_engine"],
                "tile_suggestions": {"tile_d": 128},
            },
            expected_speedup=(1.3, 5.0),
        ))

    # 3-layer MLP with bottleneck
    for suffix, D, bottleneck, FF in [
        ("mlp_3layer_bottleneck_768", 768, 256, 3072),
        ("mlp_3layer_bottleneck_1024", 1024, 256, 4096),
    ]:
        model_code = (
            "import torch\n"
            "import torch.nn as nn\n"
            "import torch.nn.functional as F\n\n"
            "class Model(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            f"        self.ln = nn.LayerNorm({D})\n"
            f"        self.fc1 = nn.Linear({D}, {bottleneck})\n"
            f"        self.fc2 = nn.Linear({bottleneck}, {FF})\n"
            f"        self.fc3 = nn.Linear({FF}, {D})\n"
            "        self.dropout = nn.Dropout(0.1)\n\n"
            "    def forward(self, x):\n"
            "        residual = x\n"
            "        x = self.ln(x)\n"
            "        x = F.gelu(self.fc1(x))\n"
            "        x = self.dropout(F.gelu(self.fc2(x)))\n"
            "        x = self.dropout(self.fc3(x))\n"
            "        return x + residual\n"
        )
        inputs_code = (
            "import torch\n\n"
            "def get_inputs():\n"
            f"    x = torch.randn(8, 128, {D}, dtype=torch.float32)\n"
            "    return [x]\n"
        )
        tasks.append(_make_task(
            name=suffix,
            category="mlp_components",
            model_code=model_code,
            inputs_code=inputs_code,
            nki_metadata={"engine_hints": ["tensor_engine", "vector_engine"]},
            expected_speedup=(1.3, 5.0),
        ))

    # Gated MLP (LLaMA SwiGLU style) with RMSNorm
    for suffix, D, FF in [
        ("gated_mlp_rms_4096", 4096, 11008),
        ("gated_mlp_rms_2048", 2048, 5504),
    ]:
        model_code = (
            "import torch\n"
            "import torch.nn as nn\n"
            "import torch.nn.functional as F\n\n"
            "class RMSNorm(nn.Module):\n"
            f"    def __init__(self, dim={D}, eps=1e-6):\n"
            "        super().__init__()\n"
            "        self.eps = eps\n"
            "        self.weight = nn.Parameter(torch.ones(dim))\n\n"
            "    def forward(self, x):\n"
            "        variance = x.pow(2).mean(-1, keepdim=True)\n"
            "        x = x * torch.rsqrt(variance + self.eps)\n"
            "        return self.weight * x\n\n"
            "class Model(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            f"        self.norm = RMSNorm({D})\n"
            f"        self.gate = nn.Linear({D}, {FF}, bias=False)\n"
            f"        self.up = nn.Linear({D}, {FF}, bias=False)\n"
            f"        self.down = nn.Linear({FF}, {D}, bias=False)\n\n"
            "    def forward(self, x):\n"
            "        residual = x\n"
            "        x = self.norm(x)\n"
            "        return residual + self.down(F.silu(self.gate(x)) * self.up(x))\n"
        )
        inputs_code = (
            "import torch\n\n"
            "def get_inputs():\n"
            f"    x = torch.randn(2, 256, {D}, dtype=torch.float32)\n"
            "    return [x]\n"
        )
        tasks.append(_make_task(
            name=suffix,
            category="mlp_components",
            model_code=model_code,
            inputs_code=inputs_code,
            nki_metadata={"engine_hints": ["tensor_engine", "vector_engine"]},
            expected_speedup=(1.3, 6.0),
        ))

    return tasks


# ============================================================================
# Transformer components (18 tasks)
# ============================================================================

def _transformer_component_tasks() -> List[NKIBenchTask]:
    tasks: List[NKIBenchTask] = []

    # Multi-head attention (full: QKV projection + attention + output projection)
    for suffix, B, S, D, H in [
        ("mha_bert_base", 8, 128, 768, 12),
        ("mha_bert_large", 4, 128, 1024, 16),
        ("mha_gpt2_small", 4, 1024, 768, 12),
        ("mha_llama_7b", 2, 256, 4096, 32),
    ]:
        head_dim = D // H
        model_code = (
            "import torch\n"
            "import torch.nn as nn\n"
            "import torch.nn.functional as F\n"
            "import math\n\n"
            "class Model(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            f"        self.n_heads = {H}\n"
            f"        self.head_dim = {head_dim}\n"
            f"        self.q_proj = nn.Linear({D}, {D}, bias=False)\n"
            f"        self.k_proj = nn.Linear({D}, {D}, bias=False)\n"
            f"        self.v_proj = nn.Linear({D}, {D}, bias=False)\n"
            f"        self.out_proj = nn.Linear({D}, {D}, bias=False)\n"
            f"        self.scale = 1.0 / math.sqrt({head_dim})\n\n"
            "    def forward(self, x):\n"
            "        B, S, D = x.shape\n"
            "        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)\n"
            "        k = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)\n"
            "        v = self.v_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)\n"
            "        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale\n"
            "        attn = F.softmax(scores, dim=-1)\n"
            "        out = torch.matmul(attn, v)\n"
            "        out = out.transpose(1, 2).contiguous().view(B, S, D)\n"
            "        return self.out_proj(out)\n"
        )
        inputs_code = (
            "import torch\n\n"
            "def get_inputs():\n"
            f"    x = torch.randn({B}, {S}, {D}, dtype=torch.float32)\n"
            "    return [x]\n"
        )
        tasks.append(_make_task(
            name=suffix,
            category="transformer_components",
            model_code=model_code,
            inputs_code=inputs_code,
            nki_metadata={
                "engine_hints": ["tensor_engine", "vector_engine"],
                "tile_suggestions": {"tile_seq": min(S, 128), "tile_head": head_dim},
            },
            expected_speedup=(1.5, 8.0),
        ))

    # Transformer encoder layer (pre-norm: LN -> MHA -> residual -> LN -> FFN -> residual)
    for suffix, B, S, D, H, FF in [
        ("enc_layer_bert_base", 8, 128, 768, 12, 3072),
        ("enc_layer_bert_large", 4, 128, 1024, 16, 4096),
        ("enc_layer_small", 8, 64, 512, 8, 2048),
        ("enc_layer_large", 2, 256, 1024, 16, 4096),
    ]:
        head_dim = D // H
        model_code = (
            "import torch\n"
            "import torch.nn as nn\n"
            "import torch.nn.functional as F\n"
            "import math\n\n"
            "class Model(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            f"        self.n_heads = {H}\n"
            f"        self.head_dim = {head_dim}\n"
            f"        self.ln1 = nn.LayerNorm({D})\n"
            f"        self.ln2 = nn.LayerNorm({D})\n"
            f"        self.q_proj = nn.Linear({D}, {D})\n"
            f"        self.k_proj = nn.Linear({D}, {D})\n"
            f"        self.v_proj = nn.Linear({D}, {D})\n"
            f"        self.out_proj = nn.Linear({D}, {D})\n"
            f"        self.fc1 = nn.Linear({D}, {FF})\n"
            f"        self.fc2 = nn.Linear({FF}, {D})\n"
            "        self.dropout = nn.Dropout(0.1)\n"
            f"        self.scale = 1.0 / math.sqrt({head_dim})\n\n"
            "    def forward(self, x):\n"
            "        # Self-attention block\n"
            "        residual = x\n"
            "        x = self.ln1(x)\n"
            "        B, S, D = x.shape\n"
            "        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)\n"
            "        k = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)\n"
            "        v = self.v_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)\n"
            "        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale\n"
            "        attn = F.softmax(scores, dim=-1)\n"
            "        attn = self.dropout(attn)\n"
            "        out = torch.matmul(attn, v)\n"
            "        out = out.transpose(1, 2).contiguous().view(B, S, D)\n"
            "        x = residual + self.dropout(self.out_proj(out))\n"
            "        # FFN block\n"
            "        residual = x\n"
            "        x = self.ln2(x)\n"
            "        x = self.dropout(F.gelu(self.fc1(x)))\n"
            "        x = self.dropout(self.fc2(x))\n"
            "        return x + residual\n"
        )
        inputs_code = (
            "import torch\n\n"
            "def get_inputs():\n"
            f"    x = torch.randn({B}, {S}, {D}, dtype=torch.float32)\n"
            "    return [x]\n"
        )
        tasks.append(_make_task(
            name=suffix,
            category="transformer_components",
            model_code=model_code,
            inputs_code=inputs_code,
            nki_metadata={
                "engine_hints": ["tensor_engine", "vector_engine"],
                "tile_suggestions": {"tile_seq": min(S, 128)},
            },
            expected_speedup=(1.5, 8.0),
        ))

    # Transformer decoder layer (causal MHA + cross-attention + FFN)
    for suffix, B, S, D, H, FF in [
        ("dec_layer_small", 4, 128, 512, 8, 2048),
        ("dec_layer_medium", 4, 256, 768, 12, 3072),
    ]:
        head_dim = D // H
        model_code = (
            "import torch\n"
            "import torch.nn as nn\n"
            "import torch.nn.functional as F\n"
            "import math\n\n"
            "class Model(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            f"        self.n_heads = {H}\n"
            f"        self.head_dim = {head_dim}\n"
            f"        self.ln1 = nn.LayerNorm({D})\n"
            f"        self.ln2 = nn.LayerNorm({D})\n"
            f"        self.ln3 = nn.LayerNorm({D})\n"
            "        # Self-attention\n"
            f"        self.self_q = nn.Linear({D}, {D})\n"
            f"        self.self_k = nn.Linear({D}, {D})\n"
            f"        self.self_v = nn.Linear({D}, {D})\n"
            f"        self.self_out = nn.Linear({D}, {D})\n"
            "        # Cross-attention\n"
            f"        self.cross_q = nn.Linear({D}, {D})\n"
            f"        self.cross_k = nn.Linear({D}, {D})\n"
            f"        self.cross_v = nn.Linear({D}, {D})\n"
            f"        self.cross_out = nn.Linear({D}, {D})\n"
            "        # FFN\n"
            f"        self.fc1 = nn.Linear({D}, {FF})\n"
            f"        self.fc2 = nn.Linear({FF}, {D})\n"
            "        self.dropout = nn.Dropout(0.1)\n"
            f"        self.scale = 1.0 / math.sqrt({head_dim})\n\n"
            "    def _attention(self, q, k, v, mask=None):\n"
            "        B, S, D = q.shape\n"
            "        q = q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)\n"
            "        k = k.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)\n"
            "        v = v.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)\n"
            "        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale\n"
            "        if mask is not None:\n"
            "            scores = scores.masked_fill(mask, float('-inf'))\n"
            "        attn = F.softmax(scores, dim=-1)\n"
            "        out = torch.matmul(attn, v)\n"
            "        return out.transpose(1, 2).contiguous().view(B, S, D)\n\n"
            "    def forward(self, x, encoder_out):\n"
            f"        S = x.shape[1]\n"
            "        # Causal self-attention\n"
            "        residual = x\n"
            "        x = self.ln1(x)\n"
            "        causal_mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()\n"
            "        q, k, v = self.self_q(x), self.self_k(x), self.self_v(x)\n"
            "        x = residual + self.dropout(self.self_out(self._attention(q, k, v, causal_mask)))\n"
            "        # Cross-attention\n"
            "        residual = x\n"
            "        x = self.ln2(x)\n"
            "        q = self.cross_q(x)\n"
            "        k, v = self.cross_k(encoder_out), self.cross_v(encoder_out)\n"
            "        x = residual + self.dropout(self.cross_out(self._attention(q, k, v)))\n"
            "        # FFN\n"
            "        residual = x\n"
            "        x = self.ln3(x)\n"
            "        x = self.dropout(F.gelu(self.fc1(x)))\n"
            "        x = self.dropout(self.fc2(x))\n"
            "        return x + residual\n"
        )
        inputs_code = (
            "import torch\n\n"
            "def get_inputs():\n"
            f"    x = torch.randn({B}, {S}, {D}, dtype=torch.float32)\n"
            f"    encoder_out = torch.randn({B}, {S}, {D}, dtype=torch.float32)\n"
            "    return [x, encoder_out]\n"
        )
        tasks.append(_make_task(
            name=suffix,
            category="transformer_components",
            model_code=model_code,
            inputs_code=inputs_code,
            nki_metadata={
                "engine_hints": ["tensor_engine", "vector_engine"],
            },
            expected_speedup=(1.5, 8.0),
        ))

    # GPT-2 block (causal MHA + MLP, pre-norm)
    for suffix, B, S, D, H, FF in [
        ("gpt2_block_small", 4, 256, 768, 12, 3072),
        ("gpt2_block_medium", 2, 512, 1024, 16, 4096),
        ("gpt2_block_large", 2, 1024, 1280, 20, 5120),
    ]:
        head_dim = D // H
        model_code = (
            "import torch\n"
            "import torch.nn as nn\n"
            "import torch.nn.functional as F\n"
            "import math\n\n"
            "class Model(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            f"        self.n_heads = {H}\n"
            f"        self.head_dim = {head_dim}\n"
            f"        self.ln1 = nn.LayerNorm({D})\n"
            f"        self.ln2 = nn.LayerNorm({D})\n"
            f"        self.attn_qkv = nn.Linear({D}, 3 * {D})\n"
            f"        self.attn_out = nn.Linear({D}, {D})\n"
            f"        self.fc1 = nn.Linear({D}, {FF})\n"
            f"        self.fc2 = nn.Linear({FF}, {D})\n"
            "        self.dropout = nn.Dropout(0.1)\n"
            f"        self.scale = 1.0 / math.sqrt({head_dim})\n\n"
            "    def forward(self, x):\n"
            "        B, S, D = x.shape\n"
            "        # Self-attention\n"
            "        residual = x\n"
            "        x = self.ln1(x)\n"
            "        qkv = self.attn_qkv(x).reshape(B, S, 3, self.n_heads, self.head_dim)\n"
            "        qkv = qkv.permute(2, 0, 3, 1, 4)\n"
            "        q, k, v = qkv[0], qkv[1], qkv[2]\n"
            "        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale\n"
            "        causal_mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()\n"
            "        scores = scores.masked_fill(causal_mask, float('-inf'))\n"
            "        attn = self.dropout(F.softmax(scores, dim=-1))\n"
            "        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, S, D)\n"
            "        x = residual + self.dropout(self.attn_out(out))\n"
            "        # MLP\n"
            "        residual = x\n"
            "        x = self.ln2(x)\n"
            "        x = self.dropout(F.gelu(self.fc1(x)))\n"
            "        x = self.dropout(self.fc2(x))\n"
            "        return x + residual\n"
        )
        inputs_code = (
            "import torch\n\n"
            "def get_inputs():\n"
            f"    x = torch.randn({B}, {S}, {D}, dtype=torch.float32)\n"
            "    return [x]\n"
        )
        tasks.append(_make_task(
            name=suffix,
            category="transformer_components",
            model_code=model_code,
            inputs_code=inputs_code,
            nki_metadata={
                "engine_hints": ["tensor_engine", "vector_engine"],
                "tile_suggestions": {"tile_seq": min(S, 128)},
            },
            expected_speedup=(1.5, 8.0),
        ))

    # BERT encoder block (post-norm variant)
    for suffix, B, S, D, H, FF in [
        ("bert_enc_base", 8, 128, 768, 12, 3072),
        ("bert_enc_large", 4, 128, 1024, 16, 4096),
    ]:
        head_dim = D // H
        model_code = (
            "import torch\n"
            "import torch.nn as nn\n"
            "import torch.nn.functional as F\n"
            "import math\n\n"
            "class Model(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            f"        self.n_heads = {H}\n"
            f"        self.head_dim = {head_dim}\n"
            f"        self.q_proj = nn.Linear({D}, {D})\n"
            f"        self.k_proj = nn.Linear({D}, {D})\n"
            f"        self.v_proj = nn.Linear({D}, {D})\n"
            f"        self.out_proj = nn.Linear({D}, {D})\n"
            f"        self.ln1 = nn.LayerNorm({D})\n"
            f"        self.ln2 = nn.LayerNorm({D})\n"
            f"        self.fc1 = nn.Linear({D}, {FF})\n"
            f"        self.fc2 = nn.Linear({FF}, {D})\n"
            "        self.dropout = nn.Dropout(0.1)\n"
            f"        self.scale = 1.0 / math.sqrt({head_dim})\n\n"
            "    def forward(self, x):\n"
            "        B, S, D = x.shape\n"
            "        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)\n"
            "        k = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)\n"
            "        v = self.v_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)\n"
            "        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale\n"
            "        attn = self.dropout(F.softmax(scores, dim=-1))\n"
            "        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, S, D)\n"
            "        x = self.ln1(x + self.dropout(self.out_proj(out)))\n"
            "        x = self.ln2(x + self.dropout(self.fc2(self.dropout(F.gelu(self.fc1(x))))))\n"
            "        return x\n"
        )
        inputs_code = (
            "import torch\n\n"
            "def get_inputs():\n"
            f"    x = torch.randn({B}, {S}, {D}, dtype=torch.float32)\n"
            "    return [x]\n"
        )
        tasks.append(_make_task(
            name=suffix,
            category="transformer_components",
            model_code=model_code,
            inputs_code=inputs_code,
            nki_metadata={
                "engine_hints": ["tensor_engine", "vector_engine"],
            },
            expected_speedup=(1.5, 8.0),
        ))

    # LLaMA-style block (RMSNorm + GQA + SwiGLU MLP)
    for suffix, B, S, D, H_q, H_kv, FF in [
        ("llama_block_7b", 2, 256, 4096, 32, 32, 11008),
        ("llama_block_13b", 1, 256, 5120, 40, 40, 13824),
        ("llama_block_gqa", 2, 256, 4096, 32, 8, 11008),
    ]:
        head_dim = D // H_q
        model_code = (
            "import torch\n"
            "import torch.nn as nn\n"
            "import torch.nn.functional as F\n"
            "import math\n\n"
            "class RMSNorm(nn.Module):\n"
            f"    def __init__(self, dim={D}, eps=1e-6):\n"
            "        super().__init__()\n"
            "        self.eps = eps\n"
            "        self.weight = nn.Parameter(torch.ones(dim))\n\n"
            "    def forward(self, x):\n"
            "        variance = x.pow(2).mean(-1, keepdim=True)\n"
            "        return self.weight * x * torch.rsqrt(variance + self.eps)\n\n"
            "class Model(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            f"        self.n_heads = {H_q}\n"
            f"        self.n_kv_heads = {H_kv}\n"
            f"        self.head_dim = {head_dim}\n"
            f"        self.attn_norm = RMSNorm({D})\n"
            f"        self.ffn_norm = RMSNorm({D})\n"
            f"        self.q_proj = nn.Linear({D}, {H_q * head_dim}, bias=False)\n"
            f"        self.k_proj = nn.Linear({D}, {H_kv * head_dim}, bias=False)\n"
            f"        self.v_proj = nn.Linear({D}, {H_kv * head_dim}, bias=False)\n"
            f"        self.o_proj = nn.Linear({H_q * head_dim}, {D}, bias=False)\n"
            f"        self.gate = nn.Linear({D}, {FF}, bias=False)\n"
            f"        self.up = nn.Linear({D}, {FF}, bias=False)\n"
            f"        self.down = nn.Linear({FF}, {D}, bias=False)\n"
            f"        self.scale = 1.0 / math.sqrt({head_dim})\n"
            f"        self.n_rep = {H_q} // {H_kv}\n\n"
            "    def forward(self, x):\n"
            "        B, S, D = x.shape\n"
            "        # Attention\n"
            "        residual = x\n"
            "        x = self.attn_norm(x)\n"
            "        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)\n"
            "        k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)\n"
            "        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)\n"
            "        if self.n_rep > 1:\n"
            "            k = k.repeat_interleave(self.n_rep, dim=1)\n"
            "            v = v.repeat_interleave(self.n_rep, dim=1)\n"
            "        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale\n"
            "        causal_mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()\n"
            "        scores = scores.masked_fill(causal_mask, float('-inf'))\n"
            "        attn = F.softmax(scores, dim=-1)\n"
            "        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, S, -1)\n"
            "        x = residual + self.o_proj(out)\n"
            "        # SwiGLU MLP\n"
            "        residual = x\n"
            "        x = self.ffn_norm(x)\n"
            "        return residual + self.down(F.silu(self.gate(x)) * self.up(x))\n"
        )
        inputs_code = (
            "import torch\n\n"
            "def get_inputs():\n"
            f"    x = torch.randn({B}, {S}, {D}, dtype=torch.float32)\n"
            "    return [x]\n"
        )
        tasks.append(_make_task(
            name=suffix,
            category="transformer_components",
            model_code=model_code,
            inputs_code=inputs_code,
            nki_metadata={
                "engine_hints": ["tensor_engine", "vector_engine"],
                "tile_suggestions": {"tile_seq": min(S, 128), "tile_head": head_dim},
            },
            expected_speedup=(1.5, 10.0),
        ))

    return tasks


# ============================================================================
# CNN components (10 tasks)
# ============================================================================

def _cnn_component_tasks() -> List[NKIBenchTask]:
    tasks: List[NKIBenchTask] = []

    # ResNet bottleneck block
    for suffix, c_in, bottleneck, c_out, stride, H, W in [
        ("resnet_bottleneck_s1", 256, 64, 256, 1, 56, 56),
        ("resnet_bottleneck_s2", 256, 128, 512, 2, 56, 56),
        ("resnet_bottleneck_large", 1024, 256, 1024, 1, 14, 14),
        ("resnet_bottleneck_final", 2048, 512, 2048, 1, 7, 7),
    ]:
        downsample_code = ""
        if c_in != c_out or stride != 1:
            downsample_code = (
                f"        self.downsample = nn.Sequential(\n"
                f"            nn.Conv2d({c_in}, {c_out}, 1, stride={stride}, bias=False),\n"
                f"            nn.BatchNorm2d({c_out}),\n"
                f"        )\n"
            )
        else:
            downsample_code = (
                "        self.downsample = None\n"
            )
        model_code = (
            "import torch\n"
            "import torch.nn as nn\n"
            "import torch.nn.functional as F\n\n"
            "class Model(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            f"        self.conv1 = nn.Conv2d({c_in}, {bottleneck}, 1, bias=False)\n"
            f"        self.bn1 = nn.BatchNorm2d({bottleneck})\n"
            f"        self.conv2 = nn.Conv2d({bottleneck}, {bottleneck}, 3, "
            f"stride={stride}, padding=1, bias=False)\n"
            f"        self.bn2 = nn.BatchNorm2d({bottleneck})\n"
            f"        self.conv3 = nn.Conv2d({bottleneck}, {c_out}, 1, bias=False)\n"
            f"        self.bn3 = nn.BatchNorm2d({c_out})\n"
            + downsample_code +
            "\n"
            "    def forward(self, x):\n"
            "        identity = x\n"
            "        out = F.relu(self.bn1(self.conv1(x)))\n"
            "        out = F.relu(self.bn2(self.conv2(out)))\n"
            "        out = self.bn3(self.conv3(out))\n"
            "        if self.downsample is not None:\n"
            "            identity = self.downsample(x)\n"
            "        return F.relu(out + identity)\n"
        )
        inputs_code = (
            "import torch\n\n"
            "def get_inputs():\n"
            f"    x = torch.randn(8, {c_in}, {H}, {W}, dtype=torch.float32)\n"
            "    return [x]\n"
        )
        tasks.append(_make_task(
            name=suffix,
            category="cnn_components",
            model_code=model_code,
            inputs_code=inputs_code,
            nki_metadata={
                "engine_hints": ["tensor_engine", "vector_engine"],
            },
            expected_speedup=(1.3, 6.0),
        ))

    # MobileNet inverted residual block (MobileNetV2)
    for suffix, c_in, expand, c_out, stride, H, W in [
        ("mobilenet_ir_s1", 16, 6, 24, 1, 56, 56),
        ("mobilenet_ir_s2", 24, 6, 32, 2, 56, 56),
        ("mobilenet_ir_large", 96, 6, 160, 2, 14, 14),
    ]:
        hidden = c_in * expand
        model_code = (
            "import torch\n"
            "import torch.nn as nn\n"
            "import torch.nn.functional as F\n\n"
            "class Model(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            f"        self.use_residual = ({c_in} == {c_out} and {stride} == 1)\n"
            f"        self.expand = nn.Sequential(\n"
            f"            nn.Conv2d({c_in}, {hidden}, 1, bias=False),\n"
            f"            nn.BatchNorm2d({hidden}),\n"
            f"            nn.ReLU6(inplace=True),\n"
            f"        )\n"
            f"        self.depthwise = nn.Sequential(\n"
            f"            nn.Conv2d({hidden}, {hidden}, 3, stride={stride}, "
            f"padding=1, groups={hidden}, bias=False),\n"
            f"            nn.BatchNorm2d({hidden}),\n"
            f"            nn.ReLU6(inplace=True),\n"
            f"        )\n"
            f"        self.project = nn.Sequential(\n"
            f"            nn.Conv2d({hidden}, {c_out}, 1, bias=False),\n"
            f"            nn.BatchNorm2d({c_out}),\n"
            f"        )\n\n"
            "    def forward(self, x):\n"
            "        identity = x\n"
            "        out = self.expand(x)\n"
            "        out = self.depthwise(out)\n"
            "        out = self.project(out)\n"
            "        if self.use_residual:\n"
            "            return out + identity\n"
            "        return out\n"
        )
        inputs_code = (
            "import torch\n\n"
            "def get_inputs():\n"
            f"    x = torch.randn(8, {c_in}, {H}, {W}, dtype=torch.float32)\n"
            "    return [x]\n"
        )
        tasks.append(_make_task(
            name=suffix,
            category="cnn_components",
            model_code=model_code,
            inputs_code=inputs_code,
            nki_metadata={"engine_hints": ["tensor_engine", "vector_engine"]},
            expected_speedup=(1.3, 5.0),
        ))

    # EfficientNet MBConv block with SE
    for suffix, c_in, expand, c_out, k, stride, se_ratio, H, W in [
        ("efficientnet_mbconv_s1", 32, 1, 16, 3, 1, 0.25, 112, 112),
        ("efficientnet_mbconv_s2", 16, 6, 24, 3, 2, 0.25, 112, 112),
        ("efficientnet_mbconv_k5", 40, 6, 80, 5, 2, 0.25, 28, 28),
    ]:
        hidden = c_in * expand
        se_channels = max(1, int(c_in * se_ratio))
        model_code = (
            "import torch\n"
            "import torch.nn as nn\n"
            "import torch.nn.functional as F\n\n"
            "class Model(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            f"        self.use_residual = ({c_in} == {c_out} and {stride} == 1)\n"
            f"        hidden = {hidden}\n"
            f"        # Expand\n"
            + (
                f"        self.expand_conv = nn.Conv2d({c_in}, {hidden}, 1, bias=False)\n"
                f"        self.expand_bn = nn.BatchNorm2d({hidden})\n"
                if expand != 1 else ""
            )
            + f"        # Depthwise\n"
            f"        self.dw_conv = nn.Conv2d({hidden}, {hidden}, {k}, stride={stride}, "
            f"padding={k // 2}, groups={hidden}, bias=False)\n"
            f"        self.dw_bn = nn.BatchNorm2d({hidden})\n"
            f"        # SE\n"
            f"        self.se_fc1 = nn.Conv2d({hidden}, {se_channels}, 1)\n"
            f"        self.se_fc2 = nn.Conv2d({se_channels}, {hidden}, 1)\n"
            f"        # Project\n"
            f"        self.proj_conv = nn.Conv2d({hidden}, {c_out}, 1, bias=False)\n"
            f"        self.proj_bn = nn.BatchNorm2d({c_out})\n\n"
            "    def forward(self, x):\n"
            "        identity = x\n"
            + (
                "        out = F.silu(self.expand_bn(self.expand_conv(x)))\n"
                if expand != 1 else "        out = x\n"
            )
            + "        out = F.silu(self.dw_bn(self.dw_conv(out)))\n"
            "        # SE\n"
            "        se = F.adaptive_avg_pool2d(out, 1)\n"
            "        se = F.silu(self.se_fc1(se))\n"
            "        se = torch.sigmoid(self.se_fc2(se))\n"
            "        out = out * se\n"
            "        out = self.proj_bn(self.proj_conv(out))\n"
            "        if self.use_residual:\n"
            "            return out + identity\n"
            "        return out\n"
        )
        inputs_code = (
            "import torch\n\n"
            "def get_inputs():\n"
            f"    x = torch.randn(8, {c_in}, {H}, {W}, dtype=torch.float32)\n"
            "    return [x]\n"
        )
        tasks.append(_make_task(
            name=suffix,
            category="cnn_components",
            model_code=model_code,
            inputs_code=inputs_code,
            nki_metadata={"engine_hints": ["tensor_engine", "vector_engine"]},
            expected_speedup=(1.3, 5.0),
        ))

    return tasks


# ============================================================================
# SSM / Mamba components (4 tasks)
# ============================================================================

def _ssm_component_tasks() -> List[NKIBenchTask]:
    tasks: List[NKIBenchTask] = []

    # Simplified Mamba selective scan block
    for suffix, B, L, D, D_state in [
        ("mamba_scan_small", 4, 256, 768, 16),
        ("mamba_scan_medium", 2, 512, 1024, 16),
        ("mamba_scan_large", 2, 1024, 1536, 16),
        ("mamba_scan_long", 1, 4096, 768, 16),
    ]:
        D_inner = D * 2
        model_code = (
            "import torch\n"
            "import torch.nn as nn\n"
            "import torch.nn.functional as F\n\n"
            "class Model(nn.Module):\n"
            "    \"\"\"Simplified Mamba block with selective scan.\"\"\"\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            f"        self.d_model = {D}\n"
            f"        self.d_inner = {D_inner}\n"
            f"        self.d_state = {D_state}\n"
            f"        self.norm = nn.LayerNorm({D})\n"
            f"        self.in_proj = nn.Linear({D}, {D_inner} * 2, bias=False)\n"
            f"        self.conv1d = nn.Conv1d({D_inner}, {D_inner}, 4, "
            f"padding=3, groups={D_inner})\n"
            f"        self.dt_proj = nn.Linear({D_inner}, {D_inner}, bias=True)\n"
            f"        self.A_log = nn.Parameter(torch.randn({D_inner}, {D_state}))\n"
            f"        self.D = nn.Parameter(torch.randn({D_inner}))\n"
            f"        self.out_proj = nn.Linear({D_inner}, {D}, bias=False)\n\n"
            "    def forward(self, x):\n"
            "        residual = x\n"
            "        x = self.norm(x)\n"
            "        B, L, D = x.shape\n"
            "        xz = self.in_proj(x)\n"
            "        x, z = xz.chunk(2, dim=-1)\n"
            "        # Conv\n"
            "        x = x.transpose(1, 2)\n"
            "        x = self.conv1d(x)[:, :, :L]\n"
            "        x = x.transpose(1, 2)\n"
            "        x = F.silu(x)\n"
            "        # Simplified selective scan (discretized)\n"
            "        A = -torch.exp(self.A_log)\n"
            "        dt = F.softplus(self.dt_proj(x))\n"
            "        dA = torch.exp(dt.unsqueeze(-1) * A)\n"
            "        # Simplified: direct multiplication instead of scan\n"
            "        y = x * self.D + x * dA.sum(-1)\n"
            "        y = y * F.silu(z)\n"
            "        return residual + self.out_proj(y)\n"
        )
        inputs_code = (
            "import torch\n\n"
            "def get_inputs():\n"
            f"    x = torch.randn({B}, {L}, {D}, dtype=torch.float32)\n"
            "    return [x]\n"
        )
        tasks.append(_make_task(
            name=suffix,
            category="ssm_components",
            model_code=model_code,
            inputs_code=inputs_code,
            nki_metadata={
                "engine_hints": ["tensor_engine", "vector_engine"],
                "tile_suggestions": {"tile_seq": min(L, 256)},
            },
            expected_speedup=(1.3, 6.0),
        ))

    return tasks


# ============================================================================
# Vision components (8 tasks)
# ============================================================================

def _vision_component_tasks() -> List[NKIBenchTask]:
    tasks: List[NKIBenchTask] = []

    # ViT patch embedding + positional encoding
    for suffix, img_size, patch_size, D, in_channels in [
        ("vit_patch_embed_224_16", 224, 16, 768, 3),
        ("vit_patch_embed_384_16", 384, 16, 1024, 3),
    ]:
        n_patches = (img_size // patch_size) ** 2
        model_code = (
            "import torch\n"
            "import torch.nn as nn\n\n"
            "class Model(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            f"        self.patch_embed = nn.Conv2d({in_channels}, {D}, "
            f"kernel_size={patch_size}, stride={patch_size})\n"
            f"        self.cls_token = nn.Parameter(torch.randn(1, 1, {D}))\n"
            f"        self.pos_embed = nn.Parameter("
            f"torch.randn(1, {n_patches} + 1, {D}))\n"
            f"        self.norm = nn.LayerNorm({D})\n\n"
            "    def forward(self, x):\n"
            "        B = x.shape[0]\n"
            "        x = self.patch_embed(x).flatten(2).transpose(1, 2)\n"
            "        cls = self.cls_token.expand(B, -1, -1)\n"
            "        x = torch.cat([cls, x], dim=1)\n"
            "        x = x + self.pos_embed\n"
            "        return self.norm(x)\n"
        )
        inputs_code = (
            "import torch\n\n"
            "def get_inputs():\n"
            f"    x = torch.randn(8, {in_channels}, {img_size}, {img_size}, "
            f"dtype=torch.float32)\n"
            "    return [x]\n"
        )
        tasks.append(_make_task(
            name=suffix,
            category="vision_components",
            model_code=model_code,
            inputs_code=inputs_code,
            nki_metadata={"engine_hints": ["tensor_engine", "vector_engine"]},
            expected_speedup=(1.2, 4.0),
        ))

    # ViT encoder block (patch embed + self-attention + MLP)
    for suffix, B, S, D, H, FF in [
        ("vit_block_base", 8, 197, 768, 12, 3072),
        ("vit_block_large", 4, 197, 1024, 16, 4096),
    ]:
        head_dim = D // H
        model_code = (
            "import torch\n"
            "import torch.nn as nn\n"
            "import torch.nn.functional as F\n"
            "import math\n\n"
            "class Model(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            f"        self.n_heads = {H}\n"
            f"        self.head_dim = {head_dim}\n"
            f"        self.ln1 = nn.LayerNorm({D})\n"
            f"        self.ln2 = nn.LayerNorm({D})\n"
            f"        self.qkv = nn.Linear({D}, 3 * {D})\n"
            f"        self.attn_out = nn.Linear({D}, {D})\n"
            f"        self.fc1 = nn.Linear({D}, {FF})\n"
            f"        self.fc2 = nn.Linear({FF}, {D})\n"
            f"        self.scale = 1.0 / math.sqrt({head_dim})\n\n"
            "    def forward(self, x):\n"
            "        B, S, D = x.shape\n"
            "        # Self-attention\n"
            "        residual = x\n"
            "        x = self.ln1(x)\n"
            "        qkv = self.qkv(x).reshape(B, S, 3, self.n_heads, self.head_dim)\n"
            "        qkv = qkv.permute(2, 0, 3, 1, 4)\n"
            "        q, k, v = qkv[0], qkv[1], qkv[2]\n"
            "        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale\n"
            "        attn = F.softmax(scores, dim=-1)\n"
            "        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, S, D)\n"
            "        x = residual + self.attn_out(out)\n"
            "        # MLP\n"
            "        residual = x\n"
            "        x = self.ln2(x)\n"
            "        x = F.gelu(self.fc1(x))\n"
            "        x = self.fc2(x)\n"
            "        return x + residual\n"
        )
        inputs_code = (
            "import torch\n\n"
            "def get_inputs():\n"
            f"    x = torch.randn({B}, {S}, {D}, dtype=torch.float32)\n"
            "    return [x]\n"
        )
        tasks.append(_make_task(
            name=suffix,
            category="vision_components",
            model_code=model_code,
            inputs_code=inputs_code,
            nki_metadata={
                "engine_hints": ["tensor_engine", "vector_engine"],
            },
            expected_speedup=(1.5, 7.0),
        ))

    # Cross-attention block (for multi-modal models)
    for suffix, B, S_q, S_kv, D, H in [
        ("cross_attn_small", 4, 196, 77, 768, 12),
        ("cross_attn_large", 2, 256, 77, 1024, 16),
    ]:
        head_dim = D // H
        model_code = (
            "import torch\n"
            "import torch.nn as nn\n"
            "import torch.nn.functional as F\n"
            "import math\n\n"
            "class Model(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            f"        self.n_heads = {H}\n"
            f"        self.head_dim = {head_dim}\n"
            f"        self.ln = nn.LayerNorm({D})\n"
            f"        self.q_proj = nn.Linear({D}, {D})\n"
            f"        self.k_proj = nn.Linear({D}, {D})\n"
            f"        self.v_proj = nn.Linear({D}, {D})\n"
            f"        self.out_proj = nn.Linear({D}, {D})\n"
            f"        self.scale = 1.0 / math.sqrt({head_dim})\n\n"
            "    def forward(self, x, context):\n"
            "        B = x.shape[0]\n"
            "        S_q = x.shape[1]\n"
            "        S_kv = context.shape[1]\n"
            "        residual = x\n"
            "        x = self.ln(x)\n"
            "        q = self.q_proj(x).view(B, S_q, self.n_heads, self.head_dim).transpose(1, 2)\n"
            "        k = self.k_proj(context).view(B, S_kv, self.n_heads, self.head_dim).transpose(1, 2)\n"
            "        v = self.v_proj(context).view(B, S_kv, self.n_heads, self.head_dim).transpose(1, 2)\n"
            "        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale\n"
            "        attn = F.softmax(scores, dim=-1)\n"
            "        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, S_q, -1)\n"
            "        return residual + self.out_proj(out)\n"
        )
        inputs_code = (
            "import torch\n\n"
            "def get_inputs():\n"
            f"    x = torch.randn({B}, {S_q}, {D}, dtype=torch.float32)\n"
            f"    context = torch.randn({B}, {S_kv}, {D}, dtype=torch.float32)\n"
            "    return [x, context]\n"
        )
        tasks.append(_make_task(
            name=suffix,
            category="vision_components",
            model_code=model_code,
            inputs_code=inputs_code,
            nki_metadata={"engine_hints": ["tensor_engine", "vector_engine"]},
            expected_speedup=(1.4, 6.0),
        ))

    # Swin Transformer window attention
    model_code = (
        "import torch\n"
        "import torch.nn as nn\n"
        "import torch.nn.functional as F\n"
        "import math\n\n"
        "class Model(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.n_heads = 8\n"
        "        self.head_dim = 32\n"
        "        self.window_size = 7\n"
        "        self.dim = 256\n"
        "        self.ln = nn.LayerNorm(256)\n"
        "        self.qkv = nn.Linear(256, 3 * 256)\n"
        "        self.proj = nn.Linear(256, 256)\n"
        "        self.scale = 1.0 / math.sqrt(32)\n\n"
        "    def forward(self, x):\n"
        "        B, H, W, C = x.shape\n"
        "        residual = x\n"
        "        x = self.ln(x)\n"
        "        # Partition into windows\n"
        "        ws = self.window_size\n"
        "        x = x.view(B, H // ws, ws, W // ws, ws, C)\n"
        "        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws * ws, C)\n"
        "        # Window attention\n"
        "        qkv = self.qkv(x).reshape(-1, ws * ws, 3, self.n_heads, self.head_dim)\n"
        "        qkv = qkv.permute(2, 0, 3, 1, 4)\n"
        "        q, k, v = qkv[0], qkv[1], qkv[2]\n"
        "        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale\n"
        "        attn = F.softmax(scores, dim=-1)\n"
        "        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(-1, ws * ws, C)\n"
        "        x = self.proj(out)\n"
        "        # Reverse window partition\n"
        "        nH, nW = H // ws, W // ws\n"
        "        x = x.view(B, nH, nW, ws, ws, C)\n"
        "        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)\n"
        "        return x + residual\n"
    )
    inputs_code = (
        "import torch\n\n"
        "def get_inputs():\n"
        "    x = torch.randn(4, 56, 56, 256, dtype=torch.float32)\n"
        "    return [x]\n"
    )
    tasks.append(_make_task(
        name="swin_window_attn",
        category="vision_components",
        model_code=model_code,
        inputs_code=inputs_code,
        nki_metadata={"engine_hints": ["tensor_engine", "vector_engine"]},
        expected_speedup=(1.4, 6.0),
    ))

    # DINOv2-style block
    model_code = (
        "import torch\n"
        "import torch.nn as nn\n"
        "import torch.nn.functional as F\n"
        "import math\n\n"
        "class Model(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        D = 768\n"
        "        self.n_heads = 12\n"
        "        self.head_dim = 64\n"
        "        self.ln1 = nn.LayerNorm(D)\n"
        "        self.ln2 = nn.LayerNorm(D)\n"
        "        self.qkv = nn.Linear(D, 3 * D)\n"
        "        self.attn_out = nn.Linear(D, D)\n"
        "        self.fc1 = nn.Linear(D, 4 * D)\n"
        "        self.fc2 = nn.Linear(4 * D, D)\n"
        "        self.ls1 = nn.Parameter(torch.ones(D) * 1e-4)\n"
        "        self.ls2 = nn.Parameter(torch.ones(D) * 1e-4)\n"
        "        self.scale = 1.0 / math.sqrt(64)\n\n"
        "    def forward(self, x):\n"
        "        B, S, D = x.shape\n"
        "        # Attention with layer scale\n"
        "        residual = x\n"
        "        x = self.ln1(x)\n"
        "        qkv = self.qkv(x).reshape(B, S, 3, self.n_heads, self.head_dim)\n"
        "        qkv = qkv.permute(2, 0, 3, 1, 4)\n"
        "        q, k, v = qkv[0], qkv[1], qkv[2]\n"
        "        attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)\n"
        "        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, S, D)\n"
        "        x = residual + self.ls1 * self.attn_out(out)\n"
        "        # MLP with layer scale\n"
        "        residual = x\n"
        "        x = self.ln2(x)\n"
        "        x = F.gelu(self.fc1(x))\n"
        "        x = self.fc2(x)\n"
        "        return residual + self.ls2 * x\n"
    )
    inputs_code = (
        "import torch\n\n"
        "def get_inputs():\n"
        "    x = torch.randn(8, 197, 768, dtype=torch.float32)\n"
        "    return [x]\n"
    )
    tasks.append(_make_task(
        name="dinov2_block",
        category="vision_components",
        model_code=model_code,
        inputs_code=inputs_code,
        nki_metadata={"engine_hints": ["tensor_engine", "vector_engine"]},
        expected_speedup=(1.5, 7.0),
    ))

    return tasks


# ============================================================================
# Specialized components (4 tasks)
# ============================================================================

def _specialized_component_tasks() -> List[NKIBenchTask]:
    tasks: List[NKIBenchTask] = []

    # Mixture of Experts (MoE) top-k routing + expert FFN
    model_code = (
        "import torch\n"
        "import torch.nn as nn\n"
        "import torch.nn.functional as F\n\n"
        "class Model(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.n_experts = 8\n"
        "        self.top_k = 2\n"
        "        self.dim = 768\n"
        "        self.ff_dim = 3072\n"
        "        self.gate = nn.Linear(768, 8)\n"
        "        self.experts_w1 = nn.Parameter(torch.randn(8, 768, 3072))\n"
        "        self.experts_w2 = nn.Parameter(torch.randn(8, 3072, 768))\n\n"
        "    def forward(self, x):\n"
        "        B, S, D = x.shape\n"
        "        x_flat = x.view(-1, D)\n"
        "        router_logits = self.gate(x_flat)\n"
        "        routing_weights = F.softmax(router_logits, dim=-1)\n"
        "        top_weights, top_indices = torch.topk(routing_weights, self.top_k, dim=-1)\n"
        "        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)\n"
        "        # Simplified expert computation\n"
        "        output = torch.zeros_like(x_flat)\n"
        "        for k in range(self.top_k):\n"
        "            expert_idx = top_indices[:, k]\n"
        "            w = top_weights[:, k:k+1]\n"
        "            for e in range(self.n_experts):\n"
        "                mask = (expert_idx == e)\n"
        "                if mask.any():\n"
        "                    expert_input = x_flat[mask]\n"
        "                    h = F.gelu(expert_input @ self.experts_w1[e])\n"
        "                    expert_out = h @ self.experts_w2[e]\n"
        "                    output[mask] += w[mask] * expert_out\n"
        "        return output.view(B, S, D)\n"
    )
    inputs_code = (
        "import torch\n\n"
        "def get_inputs():\n"
        "    x = torch.randn(4, 128, 768, dtype=torch.float32)\n"
        "    return [x]\n"
    )
    tasks.append(_make_task(
        name="moe_block",
        category="specialized_components",
        model_code=model_code,
        inputs_code=inputs_code,
        nki_metadata={
            "engine_hints": ["tensor_engine", "vector_engine"],
        },
        expected_speedup=(1.3, 6.0),
    ))

    # Flash-decoding style KV-cache attention
    model_code = (
        "import torch\n"
        "import torch.nn as nn\n"
        "import torch.nn.functional as F\n"
        "import math\n\n"
        "class Model(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.n_heads = 32\n"
        "        self.head_dim = 128\n"
        "        self.scale = 1.0 / math.sqrt(128)\n\n"
        "    def forward(self, q, k_cache, v_cache):\n"
        "        \"\"\"Single-token decoding with KV cache.\"\"\"\n"
        "        # q: (B, 1, n_heads, head_dim) -> (B, n_heads, 1, head_dim)\n"
        "        q = q.transpose(1, 2)\n"
        "        # k_cache: (B, cache_len, n_heads, head_dim) -> (B, n_heads, cache_len, head_dim)\n"
        "        k = k_cache.transpose(1, 2)\n"
        "        v = v_cache.transpose(1, 2)\n"
        "        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale\n"
        "        attn = F.softmax(scores, dim=-1)\n"
        "        out = torch.matmul(attn, v)\n"
        "        return out.transpose(1, 2)\n"
    )
    inputs_code = (
        "import torch\n\n"
        "def get_inputs():\n"
        "    B, cache_len = 4, 2048\n"
        "    q = torch.randn(B, 1, 32, 128, dtype=torch.float32)\n"
        "    k_cache = torch.randn(B, cache_len, 32, 128, dtype=torch.float32)\n"
        "    v_cache = torch.randn(B, cache_len, 32, 128, dtype=torch.float32)\n"
        "    return [q, k_cache, v_cache]\n"
    )
    tasks.append(_make_task(
        name="kv_cache_attn",
        category="specialized_components",
        model_code=model_code,
        inputs_code=inputs_code,
        nki_metadata={
            "engine_hints": ["tensor_engine", "vector_engine"],
            "tile_suggestions": {"tile_cache": 256},
        },
        expected_speedup=(1.5, 8.0),
    ))

    # Multi-query attention (MQA)
    model_code = (
        "import torch\n"
        "import torch.nn as nn\n"
        "import torch.nn.functional as F\n"
        "import math\n\n"
        "class Model(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.n_heads = 16\n"
        "        self.head_dim = 64\n"
        "        self.dim = 1024\n"
        "        self.q_proj = nn.Linear(1024, 1024, bias=False)\n"
        "        self.k_proj = nn.Linear(1024, 64, bias=False)   # single KV head\n"
        "        self.v_proj = nn.Linear(1024, 64, bias=False)\n"
        "        self.out_proj = nn.Linear(1024, 1024, bias=False)\n"
        "        self.scale = 1.0 / math.sqrt(64)\n\n"
        "    def forward(self, x):\n"
        "        B, S, D = x.shape\n"
        "        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)\n"
        "        k = self.k_proj(x).view(B, S, 1, self.head_dim).transpose(1, 2)\n"
        "        v = self.v_proj(x).view(B, S, 1, self.head_dim).transpose(1, 2)\n"
        "        # Broadcast KV across heads\n"
        "        k = k.expand(-1, self.n_heads, -1, -1)\n"
        "        v = v.expand(-1, self.n_heads, -1, -1)\n"
        "        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale\n"
        "        attn = F.softmax(scores, dim=-1)\n"
        "        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, S, D)\n"
        "        return self.out_proj(out)\n"
    )
    inputs_code = (
        "import torch\n\n"
        "def get_inputs():\n"
        "    x = torch.randn(4, 256, 1024, dtype=torch.float32)\n"
        "    return [x]\n"
    )
    tasks.append(_make_task(
        name="mqa_block",
        category="specialized_components",
        model_code=model_code,
        inputs_code=inputs_code,
        nki_metadata={"engine_hints": ["tensor_engine", "vector_engine"]},
        expected_speedup=(1.5, 7.0),
    ))

    # Perceiver-style cross-attention with latent array
    model_code = (
        "import torch\n"
        "import torch.nn as nn\n"
        "import torch.nn.functional as F\n"
        "import math\n\n"
        "class Model(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.n_heads = 8\n"
        "        self.head_dim = 64\n"
        "        self.dim = 512\n"
        "        self.ln_latent = nn.LayerNorm(512)\n"
        "        self.ln_input = nn.LayerNorm(512)\n"
        "        self.q_proj = nn.Linear(512, 512)\n"
        "        self.k_proj = nn.Linear(512, 512)\n"
        "        self.v_proj = nn.Linear(512, 512)\n"
        "        self.out_proj = nn.Linear(512, 512)\n"
        "        self.ff_ln = nn.LayerNorm(512)\n"
        "        self.ff1 = nn.Linear(512, 2048)\n"
        "        self.ff2 = nn.Linear(2048, 512)\n"
        "        self.scale = 1.0 / math.sqrt(64)\n\n"
        "    def forward(self, latents, inputs):\n"
        "        B = latents.shape[0]\n"
        "        S_q = latents.shape[1]\n"
        "        S_kv = inputs.shape[1]\n"
        "        # Cross-attention\n"
        "        residual = latents\n"
        "        latents = self.ln_latent(latents)\n"
        "        inputs_n = self.ln_input(inputs)\n"
        "        q = self.q_proj(latents).view(B, S_q, self.n_heads, self.head_dim).transpose(1, 2)\n"
        "        k = self.k_proj(inputs_n).view(B, S_kv, self.n_heads, self.head_dim).transpose(1, 2)\n"
        "        v = self.v_proj(inputs_n).view(B, S_kv, self.n_heads, self.head_dim).transpose(1, 2)\n"
        "        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale\n"
        "        attn = F.softmax(scores, dim=-1)\n"
        "        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, S_q, -1)\n"
        "        latents = residual + self.out_proj(out)\n"
        "        # FFN\n"
        "        residual = latents\n"
        "        latents = self.ff_ln(latents)\n"
        "        latents = F.gelu(self.ff1(latents))\n"
        "        latents = self.ff2(latents)\n"
        "        return latents + residual\n"
    )
    inputs_code = (
        "import torch\n\n"
        "def get_inputs():\n"
        "    latents = torch.randn(4, 64, 512, dtype=torch.float32)\n"
        "    inputs = torch.randn(4, 1024, 512, dtype=torch.float32)\n"
        "    return [latents, inputs]\n"
    )
    tasks.append(_make_task(
        name="perceiver_cross_attn",
        category="specialized_components",
        model_code=model_code,
        inputs_code=inputs_code,
        nki_metadata={"engine_hints": ["tensor_engine", "vector_engine"]},
        expected_speedup=(1.5, 7.0),
    ))

    return tasks


# ============================================================================
# Public API
# ============================================================================

def get_level3_tasks() -> List[NKIBenchTask]:
    """Return all 50 Level 3 tasks."""
    all_tasks = (
        _mlp_component_tasks()
        + _transformer_component_tasks()
        + _cnn_component_tasks()
        + _ssm_component_tasks()
        + _vision_component_tasks()
        + _specialized_component_tasks()
    )
    assert len(all_tasks) == 50, (
        f"Expected 50 Level 3 tasks, got {len(all_tasks)}"
    )
    return all_tasks
