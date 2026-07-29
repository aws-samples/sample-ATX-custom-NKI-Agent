"""
NKIGen-Bench Level 2 tasks -- fused operations (100 tasks).

Each task combines two or more PyTorch operations that should be fused into
a single NKI kernel for optimal performance.  Categories include:

  conv_fusions, matmul_fusions, norm_fusions, attention_patterns,
  mlp_blocks, residual_patterns, depthwise_separable, misc_fusions
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
    expected_speedup: tuple[float, float] = (1.2, 6.0),
) -> NKIBenchTask:
    global _COUNTER
    _COUNTER += 1
    task_id = f"L2_{category}_{_COUNTER:03d}"
    return NKIBenchTask(
        task_id=task_id,
        name=name,
        level=2,
        category=category,
        model_class_code=model_code,
        get_inputs_code=inputs_code,
        nki_metadata=nki_metadata or {},
        expected_speedup_range=expected_speedup,
    )


# ============================================================================
# Conv fusions (18 tasks)
# ============================================================================

def _conv_fusion_tasks() -> List[NKIBenchTask]:
    tasks: List[NKIBenchTask] = []

    # conv2d + relu + bias (various configs)
    conv_relu_configs = [
        ("conv2d_relu_k3", 64, 64, 3, 1, 1, 32, 32),
        ("conv2d_relu_k1", 128, 256, 1, 1, 0, 16, 16),
        ("conv2d_relu_k3_s2", 64, 128, 3, 2, 1, 64, 64),
        ("conv2d_relu_k5", 32, 64, 5, 1, 2, 32, 32),
        ("conv2d_relu_large", 3, 64, 7, 2, 3, 224, 224),
    ]
    for name, c_in, c_out, k, s, p, H, W in conv_relu_configs:
        model_code = (
            "import torch\n"
            "import torch.nn as nn\n"
            "import torch.nn.functional as F\n\n"
            "class Model(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            f"        self.conv = nn.Conv2d({c_in}, {c_out}, "
            f"kernel_size={k}, stride={s}, padding={p})\n"
            f"        self.bias = nn.Parameter(torch.randn({c_out}))\n\n"
            "    def forward(self, x):\n"
            "        x = self.conv(x)\n"
            "        x = x + self.bias.view(1, -1, 1, 1)\n"
            "        return F.relu(x)\n"
        )
        inputs_code = (
            "import torch\n\n"
            "def get_inputs():\n"
            f"    x = torch.randn(8, {c_in}, {H}, {W}, dtype=torch.float32)\n"
            "    return [x]\n"
        )
        tasks.append(_make_task(
            name=name,
            category="conv_fusions",
            model_code=model_code,
            inputs_code=inputs_code,
            nki_metadata={
                "engine_hints": ["tensor_engine", "vector_engine"],
                "tile_suggestions": {"tile_c": min(c_in, 128)},
            },
            expected_speedup=(1.3, 5.0),
        ))

    # conv2d + batch_norm + relu
    cbn_configs = [
        ("conv2d_bn_relu_k3", 64, 64, 3, 1, 1, 32, 32),
        ("conv2d_bn_relu_k1", 256, 64, 1, 1, 0, 16, 16),
        ("conv2d_bn_relu_k3_s2", 128, 256, 3, 2, 1, 32, 32),
        ("conv2d_bn_relu_large", 64, 128, 3, 1, 1, 56, 56),
        ("conv2d_bn_relu_first", 3, 64, 7, 2, 3, 224, 224),
    ]
    for name, c_in, c_out, k, s, p, H, W in cbn_configs:
        model_code = (
            "import torch\n"
            "import torch.nn as nn\n"
            "import torch.nn.functional as F\n\n"
            "class Model(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            f"        self.conv = nn.Conv2d({c_in}, {c_out}, "
            f"kernel_size={k}, stride={s}, padding={p}, bias=False)\n"
            f"        self.bn = nn.BatchNorm2d({c_out})\n\n"
            "    def forward(self, x):\n"
            "        return F.relu(self.bn(self.conv(x)))\n"
        )
        inputs_code = (
            "import torch\n\n"
            "def get_inputs():\n"
            f"    x = torch.randn(8, {c_in}, {H}, {W}, dtype=torch.float32)\n"
            "    return [x]\n"
        )
        tasks.append(_make_task(
            name=name,
            category="conv_fusions",
            model_code=model_code,
            inputs_code=inputs_code,
            nki_metadata={
                "engine_hints": ["tensor_engine", "vector_engine"],
            },
            expected_speedup=(1.3, 5.0),
        ))

    # conv2d + sigmoid (SiLU-style gate)
    model_code = (
        "import torch\n"
        "import torch.nn as nn\n\n"
        "class Model(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.conv = nn.Conv2d(64, 64, 3, 1, 1)\n\n"
        "    def forward(self, x):\n"
        "        y = self.conv(x)\n"
        "        return y * torch.sigmoid(y)\n"
    )
    inputs_code = (
        "import torch\n\n"
        "def get_inputs():\n"
        "    x = torch.randn(8, 64, 32, 32, dtype=torch.float32)\n"
        "    return [x]\n"
    )
    tasks.append(_make_task(
        name="conv2d_silu",
        category="conv_fusions",
        model_code=model_code,
        inputs_code=inputs_code,
        nki_metadata={"engine_hints": ["tensor_engine", "vector_engine"]},
        expected_speedup=(1.3, 4.0),
    ))

    # conv2d + gelu
    model_code = (
        "import torch\n"
        "import torch.nn as nn\n"
        "import torch.nn.functional as F\n\n"
        "class Model(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.conv = nn.Conv2d(128, 128, 3, 1, 1)\n\n"
        "    def forward(self, x):\n"
        "        return F.gelu(self.conv(x))\n"
    )
    inputs_code = (
        "import torch\n\n"
        "def get_inputs():\n"
        "    x = torch.randn(8, 128, 16, 16, dtype=torch.float32)\n"
        "    return [x]\n"
    )
    tasks.append(_make_task(
        name="conv2d_gelu",
        category="conv_fusions",
        model_code=model_code,
        inputs_code=inputs_code,
        nki_metadata={"engine_hints": ["tensor_engine", "vector_engine"]},
        expected_speedup=(1.3, 4.5),
    ))

    # conv2d + batch_norm + silu
    model_code = (
        "import torch\n"
        "import torch.nn as nn\n"
        "import torch.nn.functional as F\n\n"
        "class Model(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.conv = nn.Conv2d(64, 64, 3, 1, 1, bias=False)\n"
        "        self.bn = nn.BatchNorm2d(64)\n\n"
        "    def forward(self, x):\n"
        "        y = self.bn(self.conv(x))\n"
        "        return y * torch.sigmoid(y)\n"
    )
    inputs_code = (
        "import torch\n\n"
        "def get_inputs():\n"
        "    x = torch.randn(8, 64, 32, 32, dtype=torch.float32)\n"
        "    return [x]\n"
    )
    tasks.append(_make_task(
        name="conv2d_bn_silu",
        category="conv_fusions",
        model_code=model_code,
        inputs_code=inputs_code,
        nki_metadata={"engine_hints": ["tensor_engine", "vector_engine"]},
        expected_speedup=(1.3, 5.0),
    ))

    # conv2d + bn + relu + avgpool (classifier head style)
    model_code = (
        "import torch\n"
        "import torch.nn as nn\n"
        "import torch.nn.functional as F\n\n"
        "class Model(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.conv = nn.Conv2d(512, 512, 3, 1, 1, bias=False)\n"
        "        self.bn = nn.BatchNorm2d(512)\n"
        "        self.pool = nn.AdaptiveAvgPool2d((1, 1))\n\n"
        "    def forward(self, x):\n"
        "        return self.pool(F.relu(self.bn(self.conv(x))))\n"
    )
    inputs_code = (
        "import torch\n\n"
        "def get_inputs():\n"
        "    x = torch.randn(8, 512, 7, 7, dtype=torch.float32)\n"
        "    return [x]\n"
    )
    tasks.append(_make_task(
        name="conv2d_bn_relu_avgpool",
        category="conv_fusions",
        model_code=model_code,
        inputs_code=inputs_code,
        nki_metadata={"engine_hints": ["tensor_engine", "vector_engine"]},
        expected_speedup=(1.3, 4.0),
    ))

    # conv2d + relu + maxpool
    for suffix, c_in, c_out, conv_k, pool_k, H, W in [
        ("conv2d_relu_maxpool_small", 64, 64, 3, 2, 32, 32),
        ("conv2d_relu_maxpool_first", 3, 64, 7, 3, 224, 224),
        ("conv2d_relu_maxpool_mid", 128, 128, 3, 2, 28, 28),
    ]:
        model_code = (
            "import torch\n"
            "import torch.nn as nn\n"
            "import torch.nn.functional as F\n\n"
            "class Model(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            f"        self.conv = nn.Conv2d({c_in}, {c_out}, "
            f"kernel_size={conv_k}, stride=1, padding={conv_k // 2})\n"
            f"        self.pool = nn.MaxPool2d(kernel_size={pool_k}, "
            f"stride=2, padding={pool_k // 2})\n\n"
            "    def forward(self, x):\n"
            "        return self.pool(F.relu(self.conv(x)))\n"
        )
        inputs_code = (
            "import torch\n\n"
            "def get_inputs():\n"
            f"    x = torch.randn(8, {c_in}, {H}, {W}, dtype=torch.float32)\n"
            "    return [x]\n"
        )
        tasks.append(_make_task(
            name=suffix,
            category="conv_fusions",
            model_code=model_code,
            inputs_code=inputs_code,
            nki_metadata={"engine_hints": ["tensor_engine", "vector_engine"]},
            expected_speedup=(1.2, 4.0),
        ))

    # conv2d + group_norm + relu
    model_code = (
        "import torch\n"
        "import torch.nn as nn\n"
        "import torch.nn.functional as F\n\n"
        "class Model(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.conv = nn.Conv2d(256, 256, 3, 1, 1, bias=False)\n"
        "        self.gn = nn.GroupNorm(32, 256)\n\n"
        "    def forward(self, x):\n"
        "        return F.relu(self.gn(self.conv(x)))\n"
    )
    inputs_code = (
        "import torch\n\n"
        "def get_inputs():\n"
        "    x = torch.randn(8, 256, 16, 16, dtype=torch.float32)\n"
        "    return [x]\n"
    )
    tasks.append(_make_task(
        name="conv2d_gn_relu",
        category="conv_fusions",
        model_code=model_code,
        inputs_code=inputs_code,
        nki_metadata={"engine_hints": ["tensor_engine", "vector_engine"]},
        expected_speedup=(1.3, 4.5),
    ))

    return tasks


# ============================================================================
# Matmul fusions (15 tasks)
# ============================================================================

def _matmul_fusion_tasks() -> List[NKIBenchTask]:
    tasks: List[NKIBenchTask] = []

    # matmul + gelu
    for suffix, M, K, N, dtype in [
        ("matmul_gelu_small", 256, 256, 256, "torch.float32"),
        ("matmul_gelu_medium", 512, 768, 768, "torch.float32"),
        ("matmul_gelu_large", 1024, 1024, 4096, "torch.float32"),
        ("matmul_gelu_fp16", 512, 768, 768, "torch.float16"),
    ]:
        model_code = (
            "import torch\n"
            "import torch.nn as nn\n"
            "import torch.nn.functional as F\n\n"
            "class Model(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n\n"
            "    def forward(self, a, b):\n"
            "        return F.gelu(torch.matmul(a, b))\n"
        )
        inputs_code = (
            "import torch\n\n"
            "def get_inputs():\n"
            f"    a = torch.randn({M}, {K}, dtype={dtype})\n"
            f"    b = torch.randn({K}, {N}, dtype={dtype})\n"
            "    return [a, b]\n"
        )
        tasks.append(_make_task(
            name=suffix,
            category="matmul_fusions",
            model_code=model_code,
            inputs_code=inputs_code,
            nki_metadata={
                "engine_hints": ["tensor_engine", "vector_engine"],
                "tile_suggestions": {"tile_m": 128, "tile_n": 128, "tile_k": 128},
            },
            expected_speedup=(1.3, 5.0),
        ))

    # matmul + add + relu (linear layer with activation)
    for suffix, B, M, K, N in [
        ("linear_bias_relu_small", 1, 256, 256, 256),
        ("linear_bias_relu_medium", 8, 128, 768, 768),
        ("linear_bias_relu_large", 8, 128, 768, 3072),
        ("linear_bias_relu_batch", 32, 64, 512, 512),
    ]:
        model_code = (
            "import torch\n"
            "import torch.nn as nn\n"
            "import torch.nn.functional as F\n\n"
            "class Model(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            f"        self.linear = nn.Linear({K}, {N})\n\n"
            "    def forward(self, x):\n"
            "        return F.relu(self.linear(x))\n"
        )
        if B == 1:
            inputs_code = (
                "import torch\n\n"
                "def get_inputs():\n"
                f"    x = torch.randn({M}, {K}, dtype=torch.float32)\n"
                "    return [x]\n"
            )
        else:
            inputs_code = (
                "import torch\n\n"
                "def get_inputs():\n"
                f"    x = torch.randn({B}, {M}, {K}, dtype=torch.float32)\n"
                "    return [x]\n"
            )
        tasks.append(_make_task(
            name=suffix,
            category="matmul_fusions",
            model_code=model_code,
            inputs_code=inputs_code,
            nki_metadata={"engine_hints": ["tensor_engine", "vector_engine"]},
            expected_speedup=(1.2, 4.5),
        ))

    # matmul + softmax (attention score pattern)
    for suffix, B, H, S, D in [
        ("matmul_softmax_small", 1, 8, 64, 64),
        ("matmul_softmax_medium", 4, 12, 128, 64),
        ("matmul_softmax_large", 4, 16, 256, 64),
    ]:
        model_code = (
            "import torch\n"
            "import torch.nn as nn\n"
            "import torch.nn.functional as F\n"
            "import math\n\n"
            "class Model(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            f"        self.scale = 1.0 / math.sqrt({D})\n\n"
            "    def forward(self, q, k):\n"
            "        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale\n"
            "        return F.softmax(scores, dim=-1)\n"
        )
        inputs_code = (
            "import torch\n\n"
            "def get_inputs():\n"
            f"    q = torch.randn({B}, {H}, {S}, {D}, dtype=torch.float32)\n"
            f"    k = torch.randn({B}, {H}, {S}, {D}, dtype=torch.float32)\n"
            "    return [q, k]\n"
        )
        tasks.append(_make_task(
            name=suffix,
            category="matmul_fusions",
            model_code=model_code,
            inputs_code=inputs_code,
            nki_metadata={
                "engine_hints": ["tensor_engine", "vector_engine"],
                "tile_suggestions": {"tile_seq": 128},
            },
            expected_speedup=(1.5, 6.0),
        ))

    # matmul + silu (gated linear unit style)
    for suffix, M, K, N in [
        ("matmul_silu_small", 256, 768, 768),
        ("matmul_silu_medium", 512, 768, 3072),
        ("matmul_silu_large", 1024, 4096, 11008),
        ("matmul_silu_bf16", 512, 4096, 11008),
    ]:
        dtype = "torch.bfloat16" if "bf16" in suffix else "torch.float32"
        model_code = (
            "import torch\n"
            "import torch.nn as nn\n"
            "import torch.nn.functional as F\n\n"
            "class Model(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            f"        self.linear = nn.Linear({K}, {N}, bias=False)\n\n"
            "    def forward(self, x):\n"
            "        y = self.linear(x)\n"
            "        return y * torch.sigmoid(y)\n"
        )
        inputs_code = (
            "import torch\n\n"
            "def get_inputs():\n"
            f"    x = torch.randn(8, {M}, {K}, dtype={dtype})\n"
            "    return [x]\n"
        )
        tasks.append(_make_task(
            name=suffix,
            category="matmul_fusions",
            model_code=model_code,
            inputs_code=inputs_code,
            nki_metadata={"engine_hints": ["tensor_engine", "vector_engine"]},
            expected_speedup=(1.3, 5.0),
        ))

    return tasks


# ============================================================================
# Normalization fusions (10 tasks)
# ============================================================================

def _norm_fusion_tasks() -> List[NKIBenchTask]:
    tasks: List[NKIBenchTask] = []

    # layer_norm + dropout
    for suffix, shape, norm_dim, drop_p in [
        ("ln_dropout_small", "(8, 128, 768)", 768, 0.1),
        ("ln_dropout_large", "(4, 512, 1024)", 1024, 0.1),
        ("ln_dropout_gpt2", "(8, 1024, 768)", 768, 0.1),
    ]:
        model_code = (
            "import torch\n"
            "import torch.nn as nn\n\n"
            "class Model(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            f"        self.ln = nn.LayerNorm({norm_dim})\n"
            f"        self.dropout = nn.Dropout({drop_p})\n\n"
            "    def forward(self, x):\n"
            "        return self.dropout(self.ln(x))\n"
        )
        inputs_code = (
            "import torch\n\n"
            "def get_inputs():\n"
            f"    x = torch.randn({shape}, dtype=torch.float32)\n"
            "    return [x]\n"
        )
        tasks.append(_make_task(
            name=suffix,
            category="norm_fusions",
            model_code=model_code,
            inputs_code=inputs_code,
            nki_metadata={"engine_hints": ["vector_engine"]},
            expected_speedup=(1.2, 3.5),
        ))

    # rms_norm + scale
    for suffix, dim in [
        ("rms_norm_scale_768", 768),
        ("rms_norm_scale_4096", 4096),
    ]:
        model_code = (
            "import torch\n"
            "import torch.nn as nn\n\n"
            "class Model(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            f"        self.weight = nn.Parameter(torch.ones({dim}))\n"
            f"        self.scale = nn.Parameter(torch.ones({dim}))\n"
            "        self.eps = 1e-6\n\n"
            "    def forward(self, x):\n"
            "        variance = x.pow(2).mean(-1, keepdim=True)\n"
            "        x = x * torch.rsqrt(variance + self.eps)\n"
            "        return self.weight * x * self.scale\n"
        )
        inputs_code = (
            "import torch\n\n"
            "def get_inputs():\n"
            f"    x = torch.randn(8, 128, {dim}, dtype=torch.float32)\n"
            "    return [x]\n"
        )
        tasks.append(_make_task(
            name=suffix,
            category="norm_fusions",
            model_code=model_code,
            inputs_code=inputs_code,
            nki_metadata={"engine_hints": ["vector_engine"]},
            expected_speedup=(1.2, 3.5),
        ))

    # layer_norm + linear
    for suffix, seq, dim, out_dim in [
        ("ln_linear_small", 128, 768, 768),
        ("ln_linear_large", 512, 1024, 4096),
    ]:
        model_code = (
            "import torch\n"
            "import torch.nn as nn\n\n"
            "class Model(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            f"        self.ln = nn.LayerNorm({dim})\n"
            f"        self.linear = nn.Linear({dim}, {out_dim})\n\n"
            "    def forward(self, x):\n"
            "        return self.linear(self.ln(x))\n"
        )
        inputs_code = (
            "import torch\n\n"
            "def get_inputs():\n"
            f"    x = torch.randn(8, {seq}, {dim}, dtype=torch.float32)\n"
            "    return [x]\n"
        )
        tasks.append(_make_task(
            name=suffix,
            category="norm_fusions",
            model_code=model_code,
            inputs_code=inputs_code,
            nki_metadata={"engine_hints": ["vector_engine", "tensor_engine"]},
            expected_speedup=(1.3, 4.5),
        ))

    # batch_norm + relu + add (residual)
    for suffix, C, H, W in [
        ("bn_relu_add_small", 64, 56, 56),
        ("bn_relu_add_medium", 256, 14, 14),
        ("bn_relu_add_large", 512, 7, 7),
    ]:
        model_code = (
            "import torch\n"
            "import torch.nn as nn\n"
            "import torch.nn.functional as F\n\n"
            "class Model(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            f"        self.bn = nn.BatchNorm2d({C})\n\n"
            "    def forward(self, x, residual):\n"
            "        return F.relu(self.bn(x) + residual)\n"
        )
        inputs_code = (
            "import torch\n\n"
            "def get_inputs():\n"
            f"    x = torch.randn(8, {C}, {H}, {W}, dtype=torch.float32)\n"
            f"    residual = torch.randn(8, {C}, {H}, {W}, dtype=torch.float32)\n"
            "    return [x, residual]\n"
        )
        tasks.append(_make_task(
            name=suffix,
            category="norm_fusions",
            model_code=model_code,
            inputs_code=inputs_code,
            nki_metadata={"engine_hints": ["vector_engine"]},
            expected_speedup=(1.2, 3.5),
        ))

    return tasks


# ============================================================================
# Attention patterns (15 tasks)
# ============================================================================

def _attention_tasks() -> List[NKIBenchTask]:
    tasks: List[NKIBenchTask] = []

    # Full Q*K^T/sqrt(d) + mask + softmax + V
    attn_configs = [
        ("attn_full_small", 1, 8, 64, 64),
        ("attn_full_medium", 4, 12, 128, 64),
        ("attn_full_large", 4, 16, 256, 64),
        ("attn_full_bert", 8, 12, 128, 64),
        ("attn_full_gpt2", 4, 12, 1024, 64),
        ("attn_full_llama_small", 2, 32, 128, 128),
    ]
    for name, B, H, S, D in attn_configs:
        model_code = (
            "import torch\n"
            "import torch.nn as nn\n"
            "import torch.nn.functional as F\n"
            "import math\n\n"
            "class Model(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            f"        self.scale = 1.0 / math.sqrt({D})\n\n"
            "    def forward(self, q, k, v, mask):\n"
            "        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale\n"
            "        scores = scores + mask\n"
            "        attn_weights = F.softmax(scores, dim=-1)\n"
            "        return torch.matmul(attn_weights, v)\n"
        )
        inputs_code = (
            "import torch\n\n"
            "def get_inputs():\n"
            f"    q = torch.randn({B}, {H}, {S}, {D}, dtype=torch.float32)\n"
            f"    k = torch.randn({B}, {H}, {S}, {D}, dtype=torch.float32)\n"
            f"    v = torch.randn({B}, {H}, {S}, {D}, dtype=torch.float32)\n"
            f"    mask = torch.zeros({B}, 1, 1, {S}, dtype=torch.float32)\n"
            "    return [q, k, v, mask]\n"
        )
        tasks.append(_make_task(
            name=name,
            category="attention_patterns",
            model_code=model_code,
            inputs_code=inputs_code,
            nki_metadata={
                "engine_hints": ["tensor_engine", "vector_engine"],
                "tile_suggestions": {"tile_seq": min(S, 128), "tile_head": D},
            },
            expected_speedup=(1.5, 8.0),
        ))

    # Causal attention (with triangular mask)
    for name, B, H, S, D in [
        ("causal_attn_small", 2, 8, 128, 64),
        ("causal_attn_medium", 4, 12, 256, 64),
        ("causal_attn_large", 2, 16, 512, 64),
        ("causal_attn_gpt", 2, 12, 1024, 64),
    ]:
        model_code = (
            "import torch\n"
            "import torch.nn as nn\n"
            "import torch.nn.functional as F\n"
            "import math\n\n"
            "class Model(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            f"        self.scale = 1.0 / math.sqrt({D})\n\n"
            "    def forward(self, q, k, v):\n"
            "        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale\n"
            f"        causal_mask = torch.triu("
            f"torch.ones({S}, {S}, device=q.device), diagonal=1).bool()\n"
            "        scores = scores.masked_fill(causal_mask, float('-inf'))\n"
            "        attn_weights = F.softmax(scores, dim=-1)\n"
            "        return torch.matmul(attn_weights, v)\n"
        )
        inputs_code = (
            "import torch\n\n"
            "def get_inputs():\n"
            f"    q = torch.randn({B}, {H}, {S}, {D}, dtype=torch.float32)\n"
            f"    k = torch.randn({B}, {H}, {S}, {D}, dtype=torch.float32)\n"
            f"    v = torch.randn({B}, {H}, {S}, {D}, dtype=torch.float32)\n"
            "    return [q, k, v]\n"
        )
        tasks.append(_make_task(
            name=name,
            category="attention_patterns",
            model_code=model_code,
            inputs_code=inputs_code,
            nki_metadata={
                "engine_hints": ["tensor_engine", "vector_engine"],
                "tile_suggestions": {"tile_seq": min(S, 128)},
            },
            expected_speedup=(1.5, 8.0),
        ))

    # Attention with dropout
    for name, B, H, S, D in [
        ("attn_dropout_small", 4, 8, 128, 64),
        ("attn_dropout_medium", 4, 12, 256, 64),
    ]:
        model_code = (
            "import torch\n"
            "import torch.nn as nn\n"
            "import torch.nn.functional as F\n"
            "import math\n\n"
            "class Model(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            f"        self.scale = 1.0 / math.sqrt({D})\n"
            "        self.dropout = nn.Dropout(0.1)\n\n"
            "    def forward(self, q, k, v):\n"
            "        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale\n"
            "        attn_weights = F.softmax(scores, dim=-1)\n"
            "        attn_weights = self.dropout(attn_weights)\n"
            "        return torch.matmul(attn_weights, v)\n"
        )
        inputs_code = (
            "import torch\n\n"
            "def get_inputs():\n"
            f"    q = torch.randn({B}, {H}, {S}, {D}, dtype=torch.float32)\n"
            f"    k = torch.randn({B}, {H}, {S}, {D}, dtype=torch.float32)\n"
            f"    v = torch.randn({B}, {H}, {S}, {D}, dtype=torch.float32)\n"
            "    return [q, k, v]\n"
        )
        tasks.append(_make_task(
            name=name,
            category="attention_patterns",
            model_code=model_code,
            inputs_code=inputs_code,
            nki_metadata={"engine_hints": ["tensor_engine", "vector_engine"]},
            expected_speedup=(1.5, 7.0),
        ))

    # GQA (grouped query attention) pattern
    for name, B, H_q, H_kv, S, D in [
        ("gqa_small", 2, 32, 8, 128, 128),
        ("gqa_large", 2, 32, 8, 512, 128),
        ("gqa_llama", 2, 32, 8, 1024, 128),
    ]:
        model_code = (
            "import torch\n"
            "import torch.nn as nn\n"
            "import torch.nn.functional as F\n"
            "import math\n\n"
            "class Model(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            f"        self.scale = 1.0 / math.sqrt({D})\n"
            f"        self.n_rep = {H_q} // {H_kv}\n\n"
            "    def forward(self, q, k, v):\n"
            "        # Expand KV heads to match Q heads\n"
            "        k = k.repeat_interleave(self.n_rep, dim=1)\n"
            "        v = v.repeat_interleave(self.n_rep, dim=1)\n"
            "        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale\n"
            "        attn_weights = F.softmax(scores, dim=-1)\n"
            "        return torch.matmul(attn_weights, v)\n"
        )
        inputs_code = (
            "import torch\n\n"
            "def get_inputs():\n"
            f"    q = torch.randn({B}, {H_q}, {S}, {D}, dtype=torch.float32)\n"
            f"    k = torch.randn({B}, {H_kv}, {S}, {D}, dtype=torch.float32)\n"
            f"    v = torch.randn({B}, {H_kv}, {S}, {D}, dtype=torch.float32)\n"
            "    return [q, k, v]\n"
        )
        tasks.append(_make_task(
            name=name,
            category="attention_patterns",
            model_code=model_code,
            inputs_code=inputs_code,
            nki_metadata={"engine_hints": ["tensor_engine", "vector_engine"]},
            expected_speedup=(1.5, 8.0),
        ))

    return tasks


# ============================================================================
# MLP blocks (10 tasks)
# ============================================================================

def _mlp_block_tasks() -> List[NKIBenchTask]:
    tasks: List[NKIBenchTask] = []

    # 2-layer MLP: linear + gelu + linear
    for suffix, B, S, D, FF in [
        ("mlp_2layer_gelu_small", 8, 128, 768, 3072),
        ("mlp_2layer_gelu_medium", 4, 256, 1024, 4096),
        ("mlp_2layer_gelu_large", 2, 512, 4096, 11008),
        ("mlp_2layer_gelu_fp16", 8, 128, 768, 3072),
    ]:
        dtype = "torch.float16" if "fp16" in suffix else "torch.float32"
        model_code = (
            "import torch\n"
            "import torch.nn as nn\n"
            "import torch.nn.functional as F\n\n"
            "class Model(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            f"        self.fc1 = nn.Linear({D}, {FF})\n"
            f"        self.fc2 = nn.Linear({FF}, {D})\n\n"
            "    def forward(self, x):\n"
            "        return self.fc2(F.gelu(self.fc1(x)))\n"
        )
        inputs_code = (
            "import torch\n\n"
            "def get_inputs():\n"
            f"    x = torch.randn({B}, {S}, {D}, dtype={dtype})\n"
            "    return [x]\n"
        )
        tasks.append(_make_task(
            name=suffix,
            category="mlp_blocks",
            model_code=model_code,
            inputs_code=inputs_code,
            nki_metadata={"engine_hints": ["tensor_engine", "vector_engine"]},
            expected_speedup=(1.3, 5.0),
        ))

    # Gated MLP (LLaMA-style): gate * silu(up) then down
    for suffix, B, S, D, FF in [
        ("gated_mlp_small", 8, 128, 768, 2048),
        ("gated_mlp_medium", 4, 256, 1024, 2816),
        ("gated_mlp_large", 2, 512, 4096, 11008),
    ]:
        model_code = (
            "import torch\n"
            "import torch.nn as nn\n"
            "import torch.nn.functional as F\n\n"
            "class Model(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            f"        self.gate = nn.Linear({D}, {FF}, bias=False)\n"
            f"        self.up = nn.Linear({D}, {FF}, bias=False)\n"
            f"        self.down = nn.Linear({FF}, {D}, bias=False)\n\n"
            "    def forward(self, x):\n"
            "        return self.down(F.silu(self.gate(x)) * self.up(x))\n"
        )
        inputs_code = (
            "import torch\n\n"
            "def get_inputs():\n"
            f"    x = torch.randn({B}, {S}, {D}, dtype=torch.float32)\n"
            "    return [x]\n"
        )
        tasks.append(_make_task(
            name=suffix,
            category="mlp_blocks",
            model_code=model_code,
            inputs_code=inputs_code,
            nki_metadata={"engine_hints": ["tensor_engine", "vector_engine"]},
            expected_speedup=(1.3, 5.0),
        ))

    # MLP with dropout
    for suffix, B, S, D, FF in [
        ("mlp_dropout_small", 8, 128, 768, 3072),
        ("mlp_dropout_large", 4, 256, 1024, 4096),
    ]:
        model_code = (
            "import torch\n"
            "import torch.nn as nn\n"
            "import torch.nn.functional as F\n\n"
            "class Model(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            f"        self.fc1 = nn.Linear({D}, {FF})\n"
            f"        self.fc2 = nn.Linear({FF}, {D})\n"
            "        self.dropout = nn.Dropout(0.1)\n\n"
            "    def forward(self, x):\n"
            "        x = self.dropout(F.gelu(self.fc1(x)))\n"
            "        return self.fc2(x)\n"
        )
        inputs_code = (
            "import torch\n\n"
            "def get_inputs():\n"
            f"    x = torch.randn({B}, {S}, {D}, dtype=torch.float32)\n"
            "    return [x]\n"
        )
        tasks.append(_make_task(
            name=suffix,
            category="mlp_blocks",
            model_code=model_code,
            inputs_code=inputs_code,
            nki_metadata={"engine_hints": ["tensor_engine", "vector_engine"]},
            expected_speedup=(1.3, 4.5),
        ))

    # MLP with residual
    model_code = (
        "import torch\n"
        "import torch.nn as nn\n"
        "import torch.nn.functional as F\n\n"
        "class Model(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.fc1 = nn.Linear(768, 3072)\n"
        "        self.fc2 = nn.Linear(3072, 768)\n\n"
        "    def forward(self, x):\n"
        "        return x + self.fc2(F.gelu(self.fc1(x)))\n"
    )
    inputs_code = (
        "import torch\n\n"
        "def get_inputs():\n"
        "    x = torch.randn(8, 128, 768, dtype=torch.float32)\n"
        "    return [x]\n"
    )
    tasks.append(_make_task(
        name="mlp_residual",
        category="mlp_blocks",
        model_code=model_code,
        inputs_code=inputs_code,
        nki_metadata={"engine_hints": ["tensor_engine", "vector_engine"]},
        expected_speedup=(1.3, 5.0),
    ))

    return tasks


# ============================================================================
# Residual patterns (10 tasks)
# ============================================================================

def _residual_tasks() -> List[NKIBenchTask]:
    tasks: List[NKIBenchTask] = []

    # Simple residual: x + F(x) with various F
    residual_configs = [
        ("residual_linear", "nn.Linear(768, 768)", "(8, 128, 768)"),
        ("residual_conv2d",
         "nn.Conv2d(64, 64, 3, 1, 1)", "(8, 64, 32, 32)"),
        ("residual_double_linear",
         "nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 768))",
         "(8, 128, 768)"),
    ]
    for name, module_init, shape in residual_configs:
        model_code = (
            "import torch\n"
            "import torch.nn as nn\n\n"
            "class Model(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            f"        self.f = {module_init}\n\n"
            "    def forward(self, x):\n"
            "        return x + self.f(x)\n"
        )
        inputs_code = (
            "import torch\n\n"
            "def get_inputs():\n"
            f"    x = torch.randn({shape}, dtype=torch.float32)\n"
            "    return [x]\n"
        )
        tasks.append(_make_task(
            name=name,
            category="residual_patterns",
            model_code=model_code,
            inputs_code=inputs_code,
            nki_metadata={"engine_hints": ["vector_engine", "tensor_engine"]},
            expected_speedup=(1.2, 3.5),
        ))

    # Pre-norm residual: x + F(LN(x))
    for suffix, D, FF in [
        ("prenorm_residual_small", 768, 3072),
        ("prenorm_residual_large", 1024, 4096),
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
            f"        self.fc2 = nn.Linear({FF}, {D})\n\n"
            "    def forward(self, x):\n"
            "        return x + self.fc2(F.gelu(self.fc1(self.ln(x))))\n"
        )
        inputs_code = (
            "import torch\n\n"
            "def get_inputs():\n"
            f"    x = torch.randn(8, 128, {D}, dtype=torch.float32)\n"
            "    return [x]\n"
        )
        tasks.append(_make_task(
            name=suffix,
            category="residual_patterns",
            model_code=model_code,
            inputs_code=inputs_code,
            nki_metadata={"engine_hints": ["tensor_engine", "vector_engine"]},
            expected_speedup=(1.3, 5.0),
        ))

    # Weighted residual: alpha * x + (1-alpha) * F(x)
    model_code = (
        "import torch\n"
        "import torch.nn as nn\n\n"
        "class Model(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.linear = nn.Linear(768, 768)\n"
        "        self.alpha = nn.Parameter(torch.tensor(0.5))\n\n"
        "    def forward(self, x):\n"
        "        return self.alpha * x + (1 - self.alpha) * self.linear(x)\n"
    )
    inputs_code = (
        "import torch\n\n"
        "def get_inputs():\n"
        "    x = torch.randn(8, 128, 768, dtype=torch.float32)\n"
        "    return [x]\n"
    )
    tasks.append(_make_task(
        name="weighted_residual",
        category="residual_patterns",
        model_code=model_code,
        inputs_code=inputs_code,
        nki_metadata={"engine_hints": ["vector_engine", "tensor_engine"]},
        expected_speedup=(1.2, 3.5),
    ))

    # Skip + scale + norm
    for suffix, D in [
        ("skip_scale_norm_768", 768),
        ("skip_scale_norm_1024", 1024),
    ]:
        model_code = (
            "import torch\n"
            "import torch.nn as nn\n\n"
            "class Model(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            f"        self.ln = nn.LayerNorm({D})\n"
            f"        self.scale = nn.Parameter(torch.ones({D}))\n\n"
            "    def forward(self, x, residual):\n"
            "        return self.ln(x * self.scale + residual)\n"
        )
        inputs_code = (
            "import torch\n\n"
            "def get_inputs():\n"
            f"    x = torch.randn(8, 128, {D}, dtype=torch.float32)\n"
            f"    residual = torch.randn(8, 128, {D}, dtype=torch.float32)\n"
            "    return [x, residual]\n"
        )
        tasks.append(_make_task(
            name=suffix,
            category="residual_patterns",
            model_code=model_code,
            inputs_code=inputs_code,
            nki_metadata={"engine_hints": ["vector_engine"]},
            expected_speedup=(1.2, 3.0),
        ))

    # Dense connection (concatenate + project)
    for suffix, D, n_inputs in [
        ("dense_concat_2", 256, 2),
        ("dense_concat_4", 128, 4),
    ]:
        model_code = (
            "import torch\n"
            "import torch.nn as nn\n\n"
            "class Model(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            f"        self.proj = nn.Linear({D * n_inputs}, {D})\n\n"
            f"    def forward(self, {', '.join(f'x{i}' for i in range(n_inputs))}):\n"
            f"        return self.proj(torch.cat("
            f"[{', '.join(f'x{i}' for i in range(n_inputs))}], dim=-1))\n"
        )
        inputs_code = (
            "import torch\n\n"
            "def get_inputs():\n"
            + "".join(
                f"    x{i} = torch.randn(8, 128, {D}, dtype=torch.float32)\n"
                for i in range(n_inputs)
            )
            + f"    return [{', '.join(f'x{i}' for i in range(n_inputs))}]\n"
        )
        tasks.append(_make_task(
            name=suffix,
            category="residual_patterns",
            model_code=model_code,
            inputs_code=inputs_code,
            nki_metadata={"engine_hints": ["tensor_engine", "dma_engine"]},
            expected_speedup=(1.1, 3.0),
        ))

    return tasks


# ============================================================================
# Depthwise separable convolutions (8 tasks)
# ============================================================================

def _depthwise_separable_tasks() -> List[NKIBenchTask]:
    tasks: List[NKIBenchTask] = []

    configs = [
        ("dw_sep_conv_small", 32, 64, 3, 1, 32, 32),
        ("dw_sep_conv_medium", 64, 128, 3, 1, 32, 32),
        ("dw_sep_conv_large", 128, 256, 3, 1, 16, 16),
        ("dw_sep_conv_s2", 64, 128, 3, 2, 32, 32),
        ("dw_sep_conv_k5", 64, 64, 5, 1, 32, 32),
        ("dw_sep_conv_mobilenet_s1", 96, 96, 3, 1, 56, 56),
        ("dw_sep_conv_mobilenet_s2", 96, 192, 3, 2, 56, 56),
    ]
    for name, c_in, c_out, k, s, H, W in configs:
        p = k // 2
        model_code = (
            "import torch\n"
            "import torch.nn as nn\n\n"
            "class Model(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            f"        self.depthwise = nn.Conv2d({c_in}, {c_in}, "
            f"kernel_size={k}, stride={s}, padding={p}, groups={c_in})\n"
            f"        self.pointwise = nn.Conv2d({c_in}, {c_out}, "
            f"kernel_size=1)\n"
            f"        self.bn1 = nn.BatchNorm2d({c_in})\n"
            f"        self.bn2 = nn.BatchNorm2d({c_out})\n\n"
            "    def forward(self, x):\n"
            "        x = self.bn1(self.depthwise(x))\n"
            "        x = torch.relu(x)\n"
            "        x = self.bn2(self.pointwise(x))\n"
            "        return torch.relu(x)\n"
        )
        inputs_code = (
            "import torch\n\n"
            "def get_inputs():\n"
            f"    x = torch.randn(8, {c_in}, {H}, {W}, dtype=torch.float32)\n"
            "    return [x]\n"
        )
        tasks.append(_make_task(
            name=name,
            category="depthwise_separable",
            model_code=model_code,
            inputs_code=inputs_code,
            nki_metadata={"engine_hints": ["tensor_engine", "vector_engine"]},
            expected_speedup=(1.2, 4.0),
        ))

    # Grouped conv + channel shuffle (ShuffleNet style)
    model_code = (
        "import torch\n"
        "import torch.nn as nn\n\n"
        "class Model(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.conv = nn.Conv2d(128, 128, 1, groups=4)\n"
        "        self.bn = nn.BatchNorm2d(128)\n"
        "        self.groups = 4\n\n"
        "    def forward(self, x):\n"
        "        x = torch.relu(self.bn(self.conv(x)))\n"
        "        B, C, H, W = x.shape\n"
        "        x = x.view(B, self.groups, C // self.groups, H, W)\n"
        "        x = x.transpose(1, 2).contiguous()\n"
        "        return x.view(B, C, H, W)\n"
    )
    inputs_code = (
        "import torch\n\n"
        "def get_inputs():\n"
        "    x = torch.randn(8, 128, 16, 16, dtype=torch.float32)\n"
        "    return [x]\n"
    )
    tasks.append(_make_task(
        name="grouped_conv_shuffle",
        category="depthwise_separable",
        model_code=model_code,
        inputs_code=inputs_code,
        nki_metadata={"engine_hints": ["tensor_engine", "dma_engine"]},
        expected_speedup=(1.2, 3.5),
    ))

    return tasks


# ============================================================================
# Misc fusions (14 tasks)
# ============================================================================

def _misc_fusion_tasks() -> List[NKIBenchTask]:
    tasks: List[NKIBenchTask] = []

    # Embedding + layer_norm
    for suffix, V, D in [
        ("embed_ln_small", 30522, 768),
        ("embed_ln_gpt2", 50257, 768),
    ]:
        model_code = (
            "import torch\n"
            "import torch.nn as nn\n\n"
            "class Model(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            f"        self.embed = nn.Embedding({V}, {D})\n"
            f"        self.ln = nn.LayerNorm({D})\n\n"
            "    def forward(self, ids):\n"
            "        return self.ln(self.embed(ids))\n"
        )
        inputs_code = (
            "import torch\n\n"
            "def get_inputs():\n"
            f"    ids = torch.randint(0, {V}, (8, 128))\n"
            "    return [ids]\n"
        )
        tasks.append(_make_task(
            name=suffix,
            category="misc_fusions",
            model_code=model_code,
            inputs_code=inputs_code,
            nki_metadata={"engine_hints": ["vector_engine", "dma_engine"]},
            expected_speedup=(1.1, 2.5),
        ))

    # Cross-entropy loss (log_softmax + nll)
    for suffix, V in [
        ("cross_entropy_small", 1000),
        ("cross_entropy_large", 50257),
    ]:
        model_code = (
            "import torch\n"
            "import torch.nn as nn\n\n"
            "class Model(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            "        self.ce = nn.CrossEntropyLoss()\n\n"
            "    def forward(self, logits, targets):\n"
            "        return self.ce(logits, targets)\n"
        )
        inputs_code = (
            "import torch\n\n"
            "def get_inputs():\n"
            f"    logits = torch.randn(32 * 128, {V}, dtype=torch.float32)\n"
            f"    targets = torch.randint(0, {V}, (32 * 128,))\n"
            "    return [logits, targets]\n"
        )
        tasks.append(_make_task(
            name=suffix,
            category="misc_fusions",
            model_code=model_code,
            inputs_code=inputs_code,
            nki_metadata={"engine_hints": ["vector_engine"]},
            expected_speedup=(1.2, 3.0),
        ))

    # Linear + layer_norm
    for suffix, D_in, D_out in [
        ("linear_ln_small", 768, 768),
        ("linear_ln_expand", 768, 3072),
    ]:
        model_code = (
            "import torch\n"
            "import torch.nn as nn\n\n"
            "class Model(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            f"        self.linear = nn.Linear({D_in}, {D_out})\n"
            f"        self.ln = nn.LayerNorm({D_out})\n\n"
            "    def forward(self, x):\n"
            "        return self.ln(self.linear(x))\n"
        )
        inputs_code = (
            "import torch\n\n"
            "def get_inputs():\n"
            f"    x = torch.randn(8, 128, {D_in}, dtype=torch.float32)\n"
            "    return [x]\n"
        )
        tasks.append(_make_task(
            name=suffix,
            category="misc_fusions",
            model_code=model_code,
            inputs_code=inputs_code,
            nki_metadata={"engine_hints": ["tensor_engine", "vector_engine"]},
            expected_speedup=(1.3, 4.0),
        ))

    # Bilinear (two inputs projected and multiplied)
    model_code = (
        "import torch\n"
        "import torch.nn as nn\n\n"
        "class Model(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.bilinear = nn.Bilinear(256, 256, 256)\n\n"
        "    def forward(self, x, y):\n"
        "        return self.bilinear(x, y)\n"
    )
    inputs_code = (
        "import torch\n\n"
        "def get_inputs():\n"
        "    x = torch.randn(32, 256, dtype=torch.float32)\n"
        "    y = torch.randn(32, 256, dtype=torch.float32)\n"
        "    return [x, y]\n"
    )
    tasks.append(_make_task(
        name="bilinear",
        category="misc_fusions",
        model_code=model_code,
        inputs_code=inputs_code,
        nki_metadata={"engine_hints": ["tensor_engine"]},
        expected_speedup=(1.1, 3.0),
    ))

    # Scaled dot product (not full attention, just score computation)
    model_code = (
        "import torch\n"
        "import torch.nn as nn\n"
        "import math\n\n"
        "class Model(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n\n"
        "    def forward(self, q, k):\n"
        "        return torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(64)\n"
    )
    inputs_code = (
        "import torch\n\n"
        "def get_inputs():\n"
        "    q = torch.randn(4, 12, 128, 64, dtype=torch.float32)\n"
        "    k = torch.randn(4, 12, 128, 64, dtype=torch.float32)\n"
        "    return [q, k]\n"
    )
    tasks.append(_make_task(
        name="scaled_dot_product",
        category="misc_fusions",
        model_code=model_code,
        inputs_code=inputs_code,
        nki_metadata={"engine_hints": ["tensor_engine"]},
        expected_speedup=(1.1, 3.0),
    ))

    # Cosine similarity
    model_code = (
        "import torch\n"
        "import torch.nn as nn\n"
        "import torch.nn.functional as F\n\n"
        "class Model(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n\n"
        "    def forward(self, x, y):\n"
        "        return F.cosine_similarity(x, y, dim=-1)\n"
    )
    inputs_code = (
        "import torch\n\n"
        "def get_inputs():\n"
        "    x = torch.randn(32, 128, 768, dtype=torch.float32)\n"
        "    y = torch.randn(32, 128, 768, dtype=torch.float32)\n"
        "    return [x, y]\n"
    )
    tasks.append(_make_task(
        name="cosine_similarity",
        category="misc_fusions",
        model_code=model_code,
        inputs_code=inputs_code,
        nki_metadata={"engine_hints": ["vector_engine"]},
        expected_speedup=(1.1, 2.5),
    ))

    # Rotary position embedding (RoPE)
    model_code = (
        "import torch\n"
        "import torch.nn as nn\n\n"
        "class Model(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n\n"
        "    def forward(self, x, cos, sin):\n"
        "        x1 = x[..., :x.shape[-1] // 2]\n"
        "        x2 = x[..., x.shape[-1] // 2:]\n"
        "        rotated = torch.cat([-x2, x1], dim=-1)\n"
        "        return x * cos + rotated * sin\n"
    )
    inputs_code = (
        "import torch\n\n"
        "def get_inputs():\n"
        "    x = torch.randn(4, 32, 128, 128, dtype=torch.float32)\n"
        "    cos = torch.randn(1, 1, 128, 128, dtype=torch.float32)\n"
        "    sin = torch.randn(1, 1, 128, 128, dtype=torch.float32)\n"
        "    return [x, cos, sin]\n"
    )
    tasks.append(_make_task(
        name="rope_embedding",
        category="misc_fusions",
        model_code=model_code,
        inputs_code=inputs_code,
        nki_metadata={"engine_hints": ["vector_engine"]},
        expected_speedup=(1.2, 3.5),
    ))

    # SwiGLU activation (used in LLaMA, PaLM)
    model_code = (
        "import torch\n"
        "import torch.nn as nn\n"
        "import torch.nn.functional as F\n\n"
        "class Model(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.w1 = nn.Linear(768, 3072, bias=False)\n"
        "        self.w2 = nn.Linear(768, 3072, bias=False)\n\n"
        "    def forward(self, x):\n"
        "        return F.silu(self.w1(x)) * self.w2(x)\n"
    )
    inputs_code = (
        "import torch\n\n"
        "def get_inputs():\n"
        "    x = torch.randn(8, 128, 768, dtype=torch.float32)\n"
        "    return [x]\n"
    )
    tasks.append(_make_task(
        name="swiglu",
        category="misc_fusions",
        model_code=model_code,
        inputs_code=inputs_code,
        nki_metadata={"engine_hints": ["tensor_engine", "vector_engine"]},
        expected_speedup=(1.3, 4.5),
    ))

    # GeGLU
    model_code = (
        "import torch\n"
        "import torch.nn as nn\n"
        "import torch.nn.functional as F\n\n"
        "class Model(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.w1 = nn.Linear(768, 3072, bias=False)\n"
        "        self.w2 = nn.Linear(768, 3072, bias=False)\n\n"
        "    def forward(self, x):\n"
        "        return F.gelu(self.w1(x)) * self.w2(x)\n"
    )
    inputs_code = (
        "import torch\n\n"
        "def get_inputs():\n"
        "    x = torch.randn(8, 128, 768, dtype=torch.float32)\n"
        "    return [x]\n"
    )
    tasks.append(_make_task(
        name="geglu",
        category="misc_fusions",
        model_code=model_code,
        inputs_code=inputs_code,
        nki_metadata={"engine_hints": ["tensor_engine", "vector_engine"]},
        expected_speedup=(1.3, 4.5),
    ))

    # Linear + softmax (output head)
    model_code = (
        "import torch\n"
        "import torch.nn as nn\n"
        "import torch.nn.functional as F\n\n"
        "class Model(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.head = nn.Linear(768, 50257, bias=False)\n\n"
        "    def forward(self, x):\n"
        "        return F.softmax(self.head(x), dim=-1)\n"
    )
    inputs_code = (
        "import torch\n\n"
        "def get_inputs():\n"
        "    x = torch.randn(8, 128, 768, dtype=torch.float32)\n"
        "    return [x]\n"
    )
    tasks.append(_make_task(
        name="linear_softmax_head",
        category="misc_fusions",
        model_code=model_code,
        inputs_code=inputs_code,
        nki_metadata={"engine_hints": ["tensor_engine", "vector_engine"]},
        expected_speedup=(1.3, 4.0),
    ))

    # Squeeze-and-Excitation (SE) block
    model_code = (
        "import torch\n"
        "import torch.nn as nn\n\n"
        "class Model(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.fc1 = nn.Linear(256, 16)\n"
        "        self.fc2 = nn.Linear(16, 256)\n\n"
        "    def forward(self, x):\n"
        "        # x: (B, C, H, W)\n"
        "        scale = x.mean(dim=(2, 3))  # (B, C)\n"
        "        scale = torch.relu(self.fc1(scale))\n"
        "        scale = torch.sigmoid(self.fc2(scale))\n"
        "        return x * scale.unsqueeze(-1).unsqueeze(-1)\n"
    )
    inputs_code = (
        "import torch\n\n"
        "def get_inputs():\n"
        "    x = torch.randn(8, 256, 16, 16, dtype=torch.float32)\n"
        "    return [x]\n"
    )
    tasks.append(_make_task(
        name="se_block",
        category="misc_fusions",
        model_code=model_code,
        inputs_code=inputs_code,
        nki_metadata={"engine_hints": ["vector_engine", "tensor_engine"]},
        expected_speedup=(1.2, 3.5),
    ))

    return tasks


# ============================================================================
# Public API
# ============================================================================

def get_level2_tasks() -> List[NKIBenchTask]:
    """Return all 100 Level 2 tasks."""
    all_tasks = (
        _conv_fusion_tasks()
        + _matmul_fusion_tasks()
        + _norm_fusion_tasks()
        + _attention_tasks()
        + _mlp_block_tasks()
        + _residual_tasks()
        + _depthwise_separable_tasks()
        + _misc_fusion_tasks()
    )
    assert len(all_tasks) == 100, (
        f"Expected 100 Level 2 tasks, got {len(all_tasks)}"
    )
    return all_tasks
