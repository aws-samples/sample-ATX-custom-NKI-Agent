"""
NKIGen-Bench Level 1 tasks -- single operations (100 tasks).

Each task isolates a single PyTorch operation and provides:
  - A ``Model`` class (nn.Module) that performs the operation.
  - A ``get_inputs`` function returning sample input tensors.

Categories covered:
  matrix_ops, activations, normalization, reductions, pooling,
  convolutions, elementwise, softmax
"""

from __future__ import annotations

from typing import List

from . import NKIBenchTask

# ============================================================================
# Helper to build a task quickly
# ============================================================================

_COUNTER = 0


def _make_task(
    name: str,
    category: str,
    model_code: str,
    inputs_code: str,
    nki_metadata: dict | None = None,
    expected_speedup: tuple[float, float] = (1.0, 5.0),
) -> NKIBenchTask:
    global _COUNTER
    _COUNTER += 1
    task_id = f"L1_{category}_{_COUNTER:03d}"
    return NKIBenchTask(
        task_id=task_id,
        name=name,
        level=1,
        category=category,
        model_class_code=model_code,
        get_inputs_code=inputs_code,
        nki_metadata=nki_metadata or {},
        expected_speedup_range=expected_speedup,
    )


# ============================================================================
# Matrix operations (13 tasks)
# ============================================================================

def _matrix_ops_tasks() -> List[NKIBenchTask]:
    tasks: List[NKIBenchTask] = []

    # --- matmul variants (10) ---
    matmul_configs = [
        ("matmul_small_fp32", 128, 128, 128, "torch.float32"),
        ("matmul_medium_fp32", 512, 512, 512, "torch.float32"),
        ("matmul_large_fp32", 1024, 1024, 1024, "torch.float32"),
        ("matmul_xlarge_fp32", 2048, 2048, 2048, "torch.float32"),
        ("matmul_tall_fp32", 2048, 128, 512, "torch.float32"),
        ("matmul_wide_fp32", 128, 2048, 512, "torch.float32"),
        ("matmul_small_fp16", 256, 256, 256, "torch.float16"),
        ("matmul_medium_fp16", 512, 512, 512, "torch.float16"),
        ("matmul_large_fp16", 1024, 1024, 1024, "torch.float16"),
        ("matmul_bf16", 1024, 1024, 1024, "torch.bfloat16"),
    ]
    for name, M, K, N, dtype in matmul_configs:
        model_code = (
            "import torch\n"
            "import torch.nn as nn\n\n"
            "class Model(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n\n"
            "    def forward(self, a, b):\n"
            "        return torch.matmul(a, b)\n"
        )
        inputs_code = (
            "import torch\n\n"
            "def get_inputs():\n"
            f"    a = torch.randn({M}, {K}, dtype={dtype})\n"
            f"    b = torch.randn({K}, {N}, dtype={dtype})\n"
            "    return [a, b]\n"
        )
        tasks.append(_make_task(
            name=name,
            category="matrix_ops",
            model_code=model_code,
            inputs_code=inputs_code,
            nki_metadata={
                "engine_hints": ["tensor_engine"],
                "tile_suggestions": {"tile_m": 128, "tile_n": 128, "tile_k": 128},
            },
            expected_speedup=(1.2, 4.0),
        ))

    # --- bmm ---
    model_code = (
        "import torch\n"
        "import torch.nn as nn\n\n"
        "class Model(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n\n"
        "    def forward(self, a, b):\n"
        "        return torch.bmm(a, b)\n"
    )
    inputs_code = (
        "import torch\n\n"
        "def get_inputs():\n"
        "    a = torch.randn(8, 256, 256, dtype=torch.float32)\n"
        "    b = torch.randn(8, 256, 256, dtype=torch.float32)\n"
        "    return [a, b]\n"
    )
    tasks.append(_make_task(
        name="bmm_fp32",
        category="matrix_ops",
        model_code=model_code,
        inputs_code=inputs_code,
        nki_metadata={"engine_hints": ["tensor_engine"]},
        expected_speedup=(1.2, 3.5),
    ))

    # --- outer product ---
    model_code = (
        "import torch\n"
        "import torch.nn as nn\n\n"
        "class Model(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n\n"
        "    def forward(self, a, b):\n"
        "        return torch.outer(a, b)\n"
    )
    inputs_code = (
        "import torch\n\n"
        "def get_inputs():\n"
        "    a = torch.randn(1024, dtype=torch.float32)\n"
        "    b = torch.randn(1024, dtype=torch.float32)\n"
        "    return [a, b]\n"
    )
    tasks.append(_make_task(
        name="outer_product_fp32",
        category="matrix_ops",
        model_code=model_code,
        inputs_code=inputs_code,
        nki_metadata={"engine_hints": ["tensor_engine"]},
        expected_speedup=(1.1, 3.0),
    ))

    # --- vector dot product ---
    model_code = (
        "import torch\n"
        "import torch.nn as nn\n\n"
        "class Model(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n\n"
        "    def forward(self, a, b):\n"
        "        return torch.dot(a, b)\n"
    )
    inputs_code = (
        "import torch\n\n"
        "def get_inputs():\n"
        "    a = torch.randn(4096, dtype=torch.float32)\n"
        "    b = torch.randn(4096, dtype=torch.float32)\n"
        "    return [a, b]\n"
    )
    tasks.append(_make_task(
        name="dot_product_fp32",
        category="matrix_ops",
        model_code=model_code,
        inputs_code=inputs_code,
        nki_metadata={"engine_hints": ["vector_engine"]},
        expected_speedup=(1.0, 2.0),
    ))

    return tasks


# ============================================================================
# Activation functions (10 tasks)
# ============================================================================

def _activation_tasks() -> List[NKIBenchTask]:
    tasks: List[NKIBenchTask] = []

    activations = [
        ("relu", "torch.nn.functional.relu(x)"),
        ("gelu", "torch.nn.functional.gelu(x)"),
        ("silu", "torch.nn.functional.silu(x)"),
        ("sigmoid", "torch.sigmoid(x)"),
        ("tanh", "torch.tanh(x)"),
        ("leaky_relu", "torch.nn.functional.leaky_relu(x, 0.01)"),
        ("elu", "torch.nn.functional.elu(x, 1.0)"),
        ("softplus", "torch.nn.functional.softplus(x)"),
        ("mish", "torch.nn.functional.mish(x)"),
        ("swish", "x * torch.sigmoid(x)"),
    ]

    for act_name, act_expr in activations:
        model_code = (
            "import torch\n"
            "import torch.nn as nn\n"
            "import torch.nn.functional as F\n\n"
            "class Model(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n\n"
            "    def forward(self, x):\n"
            f"        return {act_expr}\n"
        )
        inputs_code = (
            "import torch\n\n"
            "def get_inputs():\n"
            "    x = torch.randn(16, 512, 512, dtype=torch.float32)\n"
            "    return [x]\n"
        )
        tasks.append(_make_task(
            name=f"{act_name}_fp32",
            category="activations",
            model_code=model_code,
            inputs_code=inputs_code,
            nki_metadata={
                "engine_hints": ["vector_engine"],
                "tile_suggestions": {"tile_size": 512},
            },
            expected_speedup=(1.1, 3.0),
        ))

    return tasks


# ============================================================================
# Normalization (10 tasks: layer_norm x3, batch_norm x2, instance_norm, group_norm x2, rms_norm x2)
# ============================================================================

def _normalization_tasks() -> List[NKIBenchTask]:
    tasks: List[NKIBenchTask] = []

    # layer_norm variants
    for suffix, shape, norm_shape in [
        ("layer_norm_2d", "(32, 768)", "[768]"),
        ("layer_norm_3d", "(8, 128, 768)", "[768]"),
        ("layer_norm_large", "(4, 512, 1024)", "[1024]"),
    ]:
        model_code = (
            "import torch\n"
            "import torch.nn as nn\n\n"
            "class Model(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            f"        self.ln = nn.LayerNorm({norm_shape})\n\n"
            "    def forward(self, x):\n"
            "        return self.ln(x)\n"
        )
        inputs_code = (
            "import torch\n\n"
            "def get_inputs():\n"
            f"    x = torch.randn({shape}, dtype=torch.float32)\n"
            "    return [x]\n"
        )
        tasks.append(_make_task(
            name=suffix,
            category="normalization",
            model_code=model_code,
            inputs_code=inputs_code,
            nki_metadata={"engine_hints": ["vector_engine"]},
            expected_speedup=(1.1, 3.0),
        ))

    # batch_norm variants
    for suffix, N, C, H, W in [
        ("batch_norm_small", 8, 64, 32, 32),
        ("batch_norm_large", 16, 256, 16, 16),
    ]:
        model_code = (
            "import torch\n"
            "import torch.nn as nn\n\n"
            "class Model(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            f"        self.bn = nn.BatchNorm2d({C})\n\n"
            "    def forward(self, x):\n"
            "        return self.bn(x)\n"
        )
        inputs_code = (
            "import torch\n\n"
            "def get_inputs():\n"
            f"    x = torch.randn({N}, {C}, {H}, {W}, dtype=torch.float32)\n"
            "    return [x]\n"
        )
        tasks.append(_make_task(
            name=suffix,
            category="normalization",
            model_code=model_code,
            inputs_code=inputs_code,
            nki_metadata={"engine_hints": ["vector_engine"]},
            expected_speedup=(1.1, 2.5),
        ))

    # instance_norm
    model_code = (
        "import torch\n"
        "import torch.nn as nn\n\n"
        "class Model(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.inst_norm = nn.InstanceNorm2d(128)\n\n"
        "    def forward(self, x):\n"
        "        return self.inst_norm(x)\n"
    )
    inputs_code = (
        "import torch\n\n"
        "def get_inputs():\n"
        "    x = torch.randn(8, 128, 32, 32, dtype=torch.float32)\n"
        "    return [x]\n"
    )
    tasks.append(_make_task(
        name="instance_norm",
        category="normalization",
        model_code=model_code,
        inputs_code=inputs_code,
        nki_metadata={"engine_hints": ["vector_engine"]},
        expected_speedup=(1.1, 2.5),
    ))

    # group_norm variants
    for suffix, N, C, H, W, groups in [
        ("group_norm_g8", 8, 256, 16, 16, 8),
        ("group_norm_g32", 8, 256, 16, 16, 32),
    ]:
        model_code = (
            "import torch\n"
            "import torch.nn as nn\n\n"
            "class Model(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            f"        self.gn = nn.GroupNorm({groups}, {C})\n\n"
            "    def forward(self, x):\n"
            "        return self.gn(x)\n"
        )
        inputs_code = (
            "import torch\n\n"
            "def get_inputs():\n"
            f"    x = torch.randn({N}, {C}, {H}, {W}, dtype=torch.float32)\n"
            "    return [x]\n"
        )
        tasks.append(_make_task(
            name=suffix,
            category="normalization",
            model_code=model_code,
            inputs_code=inputs_code,
            nki_metadata={"engine_hints": ["vector_engine"]},
            expected_speedup=(1.1, 2.5),
        ))

    # rms_norm
    model_code = (
        "import torch\n"
        "import torch.nn as nn\n\n"
        "class Model(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.weight = nn.Parameter(torch.ones(768))\n"
        "        self.eps = 1e-6\n\n"
        "    def forward(self, x):\n"
        "        variance = x.pow(2).mean(-1, keepdim=True)\n"
        "        x = x * torch.rsqrt(variance + self.eps)\n"
        "        return self.weight * x\n"
    )
    inputs_code = (
        "import torch\n\n"
        "def get_inputs():\n"
        "    x = torch.randn(8, 128, 768, dtype=torch.float32)\n"
        "    return [x]\n"
    )
    tasks.append(_make_task(
        name="rms_norm",
        category="normalization",
        model_code=model_code,
        inputs_code=inputs_code,
        nki_metadata={"engine_hints": ["vector_engine"]},
        expected_speedup=(1.2, 3.5),
    ))

    # rms_norm large (4096-dim, LLaMA-style)
    model_code = (
        "import torch\n"
        "import torch.nn as nn\n\n"
        "class Model(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.weight = nn.Parameter(torch.ones(4096))\n"
        "        self.eps = 1e-6\n\n"
        "    def forward(self, x):\n"
        "        variance = x.pow(2).mean(-1, keepdim=True)\n"
        "        x = x * torch.rsqrt(variance + self.eps)\n"
        "        return self.weight * x\n"
    )
    inputs_code = (
        "import torch\n\n"
        "def get_inputs():\n"
        "    x = torch.randn(4, 512, 4096, dtype=torch.float32)\n"
        "    return [x]\n"
    )
    tasks.append(_make_task(
        name="rms_norm_large",
        category="normalization",
        model_code=model_code,
        inputs_code=inputs_code,
        nki_metadata={"engine_hints": ["vector_engine"]},
        expected_speedup=(1.2, 3.5),
    ))

    return tasks


# ============================================================================
# Reductions (12 tasks)
# ============================================================================

def _reduction_tasks() -> List[NKIBenchTask]:
    tasks: List[NKIBenchTask] = []

    reduction_configs = [
        ("sum_axis0", "torch.sum(x, dim=0)", "(1024, 512)"),
        ("sum_axis1", "torch.sum(x, dim=1)", "(1024, 512)"),
        ("sum_all", "torch.sum(x)", "(1024, 512)"),
        ("mean_axis0", "torch.mean(x, dim=0)", "(1024, 512)"),
        ("mean_axis1", "torch.mean(x, dim=1)", "(1024, 512)"),
        ("mean_3d_axis2", "torch.mean(x, dim=2)", "(8, 128, 768)"),
        ("max_axis0", "torch.max(x, dim=0)[0]", "(1024, 512)"),
        ("max_axis1", "torch.max(x, dim=1)[0]", "(1024, 512)"),
        ("min_axis0", "torch.min(x, dim=0)[0]", "(1024, 512)"),
        ("min_axis1", "torch.min(x, dim=1)[0]", "(1024, 512)"),
        ("sum_large_axis0", "torch.sum(x, dim=0)", "(4096, 1024)"),
        ("mean_large_axis1", "torch.mean(x, dim=1)", "(4096, 1024)"),
    ]

    for name, op_expr, shape in reduction_configs:
        model_code = (
            "import torch\n"
            "import torch.nn as nn\n\n"
            "class Model(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n\n"
            "    def forward(self, x):\n"
            f"        return {op_expr}\n"
        )
        inputs_code = (
            "import torch\n\n"
            "def get_inputs():\n"
            f"    x = torch.randn({shape}, dtype=torch.float32)\n"
            "    return [x]\n"
        )
        tasks.append(_make_task(
            name=name,
            category="reductions",
            model_code=model_code,
            inputs_code=inputs_code,
            nki_metadata={"engine_hints": ["vector_engine"]},
            expected_speedup=(1.0, 2.5),
        ))

    return tasks


# ============================================================================
# Pooling (7 tasks)
# ============================================================================

def _pooling_tasks() -> List[NKIBenchTask]:
    tasks: List[NKIBenchTask] = []

    pooling_configs = [
        ("avg_pool2d_k3", "nn.AvgPool2d(kernel_size=3, stride=1, padding=1)",
         8, 64, 32, 32),
        ("avg_pool2d_k5", "nn.AvgPool2d(kernel_size=5, stride=2, padding=2)",
         8, 64, 64, 64),
        ("max_pool2d_k2_s2", "nn.MaxPool2d(kernel_size=2, stride=2)",
         8, 128, 32, 32),
        ("max_pool2d_k3_s2", "nn.MaxPool2d(kernel_size=3, stride=2, padding=1)",
         8, 128, 64, 64),
        ("adaptive_avg_pool_7", "nn.AdaptiveAvgPool2d((7, 7))",
         8, 512, 14, 14),
        ("adaptive_avg_pool_1", "nn.AdaptiveAvgPool2d((1, 1))",
         8, 2048, 7, 7),
        ("max_pool2d_large", "nn.MaxPool2d(kernel_size=3, stride=2, padding=1)",
         16, 64, 112, 112),
    ]

    for name, pool_init, N, C, H, W in pooling_configs:
        model_code = (
            "import torch\n"
            "import torch.nn as nn\n\n"
            "class Model(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            f"        self.pool = {pool_init}\n\n"
            "    def forward(self, x):\n"
            "        return self.pool(x)\n"
        )
        inputs_code = (
            "import torch\n\n"
            "def get_inputs():\n"
            f"    x = torch.randn({N}, {C}, {H}, {W}, dtype=torch.float32)\n"
            "    return [x]\n"
        )
        tasks.append(_make_task(
            name=name,
            category="pooling",
            model_code=model_code,
            inputs_code=inputs_code,
            nki_metadata={"engine_hints": ["vector_engine"]},
            expected_speedup=(1.0, 2.5),
        ))

    return tasks


# ============================================================================
# Convolutions (14 tasks)
# ============================================================================

def _convolution_tasks() -> List[NKIBenchTask]:
    tasks: List[NKIBenchTask] = []

    # conv1d
    conv1d_configs = [
        ("conv1d_k3", 64, 64, 3, 1, 1, 256),
        ("conv1d_k5", 64, 128, 5, 1, 2, 256),
        ("conv1d_k7_s2", 64, 128, 7, 2, 3, 512),
    ]
    for name, c_in, c_out, k, s, p, length in conv1d_configs:
        model_code = (
            "import torch\n"
            "import torch.nn as nn\n\n"
            "class Model(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            f"        self.conv = nn.Conv1d({c_in}, {c_out}, "
            f"kernel_size={k}, stride={s}, padding={p})\n\n"
            "    def forward(self, x):\n"
            "        return self.conv(x)\n"
        )
        inputs_code = (
            "import torch\n\n"
            "def get_inputs():\n"
            f"    x = torch.randn(8, {c_in}, {length}, dtype=torch.float32)\n"
            "    return [x]\n"
        )
        tasks.append(_make_task(
            name=name,
            category="convolutions",
            model_code=model_code,
            inputs_code=inputs_code,
            nki_metadata={
                "engine_hints": ["tensor_engine"],
                "tile_suggestions": {"tile_c": 64},
            },
            expected_speedup=(1.1, 3.0),
        ))

    # conv2d
    conv2d_configs = [
        ("conv2d_k1", 64, 128, 1, 1, 0, 32, 32),
        ("conv2d_k3", 64, 64, 3, 1, 1, 32, 32),
        ("conv2d_k3_s2", 64, 128, 3, 2, 1, 64, 64),
        ("conv2d_k5", 32, 64, 5, 1, 2, 32, 32),
        ("conv2d_k7_s2", 3, 64, 7, 2, 3, 224, 224),
        ("conv2d_large_cin", 512, 512, 3, 1, 1, 8, 8),
        ("conv2d_depthwise", 128, 128, 3, 1, 1, 32, 32),
        ("conv2d_k3_fp16", 64, 64, 3, 1, 1, 32, 32),
        ("conv2d_k1_bottleneck", 256, 64, 1, 1, 0, 16, 16),
        ("conv2d_k3_dilated", 64, 64, 3, 1, 2, 32, 32),
        ("conv2d_large_spatial", 64, 64, 3, 1, 1, 112, 112),
    ]
    for i, (name, c_in, c_out, k, s, p, H, W) in enumerate(conv2d_configs):
        groups = c_in if name == "conv2d_depthwise" else 1
        dilation = 2 if "dilated" in name else 1
        dtype_str = "torch.float16" if "fp16" in name else "torch.float32"
        model_code = (
            "import torch\n"
            "import torch.nn as nn\n\n"
            "class Model(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            f"        self.conv = nn.Conv2d({c_in}, {c_out}, "
            f"kernel_size={k}, stride={s}, padding={p}, "
            f"dilation={dilation}, groups={groups})\n\n"
            "    def forward(self, x):\n"
            "        return self.conv(x)\n"
        )
        inputs_code = (
            "import torch\n\n"
            "def get_inputs():\n"
            f"    x = torch.randn(8, {c_in}, {H}, {W}, dtype={dtype_str})\n"
            "    return [x]\n"
        )
        tasks.append(_make_task(
            name=name,
            category="convolutions",
            model_code=model_code,
            inputs_code=inputs_code,
            nki_metadata={
                "engine_hints": ["tensor_engine"],
                "tile_suggestions": {"tile_c": min(c_in, 128)},
            },
            expected_speedup=(1.1, 3.5),
        ))

    return tasks


# ============================================================================
# Elementwise operations (16 tasks)
# ============================================================================

def _elementwise_tasks() -> List[NKIBenchTask]:
    tasks: List[NKIBenchTask] = []

    # Binary ops
    binary_ops = [
        ("add", "x + y"),
        ("mul", "x * y"),
        ("sub", "x - y"),
        ("div", "x / (y + 1e-6)"),
    ]
    for name, expr in binary_ops:
        model_code = (
            "import torch\n"
            "import torch.nn as nn\n\n"
            "class Model(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n\n"
            "    def forward(self, x, y):\n"
            f"        return {expr}\n"
        )
        inputs_code = (
            "import torch\n\n"
            "def get_inputs():\n"
            "    x = torch.randn(16, 512, 512, dtype=torch.float32)\n"
            "    y = torch.randn(16, 512, 512, dtype=torch.float32)\n"
            "    return [x, y]\n"
        )
        tasks.append(_make_task(
            name=f"elem_{name}",
            category="elementwise",
            model_code=model_code,
            inputs_code=inputs_code,
            nki_metadata={"engine_hints": ["vector_engine"]},
            expected_speedup=(1.0, 2.0),
        ))

    # Unary ops
    unary_ops = [
        ("exp", "torch.exp(x)"),
        ("log", "torch.log(torch.abs(x) + 1e-6)"),
        ("sqrt", "torch.sqrt(torch.abs(x))"),
        ("rsqrt", "torch.rsqrt(torch.abs(x) + 1e-6)"),
        ("abs", "torch.abs(x)"),
        ("neg", "torch.neg(x)"),
        ("sin", "torch.sin(x)"),
        ("cos", "torch.cos(x)"),
        ("reciprocal", "torch.reciprocal(x + 1e-6)"),
        ("square", "x * x"),
        ("clamp", "torch.clamp(x, -1.0, 1.0)"),
        ("pow2", "torch.pow(x, 2)"),
    ]
    for name, expr in unary_ops:
        model_code = (
            "import torch\n"
            "import torch.nn as nn\n\n"
            "class Model(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n\n"
            "    def forward(self, x):\n"
            f"        return {expr}\n"
        )
        inputs_code = (
            "import torch\n\n"
            "def get_inputs():\n"
            "    x = torch.randn(16, 512, 512, dtype=torch.float32)\n"
            "    return [x]\n"
        )
        tasks.append(_make_task(
            name=f"elem_{name}",
            category="elementwise",
            model_code=model_code,
            inputs_code=inputs_code,
            nki_metadata={"engine_hints": ["vector_engine"]},
            expected_speedup=(1.0, 2.0),
        ))

    return tasks


# ============================================================================
# Softmax (8 tasks)
# ============================================================================

def _softmax_tasks() -> List[NKIBenchTask]:
    tasks: List[NKIBenchTask] = []

    softmax_configs = [
        ("softmax_dim1_2d", "F.softmax(x, dim=1)", "(1024, 512)"),
        ("softmax_dim0_2d", "F.softmax(x, dim=0)", "(1024, 512)"),
        ("softmax_dim2_3d", "F.softmax(x, dim=2)", "(8, 128, 768)"),
        ("softmax_dim1_3d", "F.softmax(x, dim=1)", "(8, 128, 768)"),
        ("softmax_large", "F.softmax(x, dim=-1)", "(32, 128, 50257)"),
        ("log_softmax_dim1", "F.log_softmax(x, dim=1)", "(1024, 512)"),
        ("log_softmax_dim2", "F.log_softmax(x, dim=2)", "(8, 128, 768)"),
        ("log_softmax_large", "F.log_softmax(x, dim=-1)", "(32, 128, 50257)"),
    ]

    for name, op_expr, shape in softmax_configs:
        model_code = (
            "import torch\n"
            "import torch.nn as nn\n"
            "import torch.nn.functional as F\n\n"
            "class Model(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n\n"
            "    def forward(self, x):\n"
            f"        return {op_expr}\n"
        )
        inputs_code = (
            "import torch\n\n"
            "def get_inputs():\n"
            f"    x = torch.randn({shape}, dtype=torch.float32)\n"
            "    return [x]\n"
        )
        tasks.append(_make_task(
            name=name,
            category="softmax",
            model_code=model_code,
            inputs_code=inputs_code,
            nki_metadata={
                "engine_hints": ["vector_engine"],
                "tile_suggestions": {"tile_size": 128},
            },
            expected_speedup=(1.2, 4.0),
        ))

    return tasks


# ============================================================================
# Transpose / Reshape / Gather (10 tasks)
# ============================================================================

def _data_movement_tasks() -> List[NKIBenchTask]:
    tasks: List[NKIBenchTask] = []

    configs = [
        ("transpose_2d", "x.T", "(1024, 512)"),
        ("permute_3d", "x.permute(0, 2, 1)", "(8, 128, 768)"),
        ("permute_4d", "x.permute(0, 2, 3, 1)", "(8, 64, 32, 32)"),
        ("contiguous_after_transpose", "x.transpose(1, 2).contiguous()",
         "(8, 128, 768)"),
        ("reshape_flatten", "x.reshape(x.shape[0], -1)", "(8, 64, 32, 32)"),
        ("cat_dim0", "torch.cat([x, x], dim=0)", "(512, 512)"),
        ("cat_dim1", "torch.cat([x, x], dim=1)", "(512, 512)"),
        ("stack_dim0", "torch.stack([x, x], dim=0)", "(512, 512)"),
        ("gather_axis1",
         "torch.gather(x, 1, torch.randint(0, 512, (1024, 256)).to(x.device))",
         "(1024, 512)"),
        ("index_select",
         "torch.index_select(x, 0, torch.arange(0, 512, 2).to(x.device))",
         "(1024, 512)"),
    ]

    for name, op_expr, shape in configs:
        model_code = (
            "import torch\n"
            "import torch.nn as nn\n\n"
            "class Model(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n\n"
            "    def forward(self, x):\n"
            f"        return {op_expr}\n"
        )
        inputs_code = (
            "import torch\n\n"
            "def get_inputs():\n"
            f"    x = torch.randn({shape}, dtype=torch.float32)\n"
            "    return [x]\n"
        )
        tasks.append(_make_task(
            name=name,
            category="data_movement",
            model_code=model_code,
            inputs_code=inputs_code,
            nki_metadata={"engine_hints": ["dma_engine"]},
            expected_speedup=(1.0, 2.0),
        ))

    return tasks


# ============================================================================
# Public API
# ============================================================================

def get_level1_tasks() -> List[NKIBenchTask]:
    """Return all 100 Level 1 tasks."""
    all_tasks = (
        _matrix_ops_tasks()
        + _activation_tasks()
        + _normalization_tasks()
        + _reduction_tasks()
        + _pooling_tasks()
        + _convolution_tasks()
        + _elementwise_tasks()
        + _softmax_tasks()
        + _data_movement_tasks()
    )
    assert len(all_tasks) == 100, (
        f"Expected 100 Level 1 tasks, got {len(all_tasks)}"
    )
    return all_tasks
