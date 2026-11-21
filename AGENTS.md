# ComfyUI - Agent Instructions

## Project Overview

ComfyUI is a node/graph-based UI for Stable Diffusion and other AI models. It uses a flowchart interface where nodes represent operations and edges represent data flow. The project follows a weekly release cycle.

## Architecture

- **`main.py`** - Entry point for the ComfyUI application
- **`comfy/`** - Core engine: node execution, model loading, workflow processing
- **`comfy_execution/`** - Execution pipeline and graph processing
- **`comfy_api/`** - API nodes for external model providers
- **`api_server/`** - REST API server implementation
- **`app/`** - Web application and server logic
- **`custom_nodes/`** - Extension system for third-party nodes
- **`user/default/workflows/`** - Custom workflows directory (our target for new workflows)
- **`web/`** - Compiled frontend from separate repo [ComfyUI_frontend](https://github.com/Comfy-Org/ComfyUI_frontend)
- **`models/`** - Model storage directory (checkpoints, VAE, LoRA, etc.)
- **`tests/`** and **`tests-unit/`** - Test suites

## Python Environment

**Virtual environment**: `.venv/` (at project root `/media/abdelrahman/Data/ComfyUI/.venv/`)

**Package manager**: [uv](https://docs.astral.sh/uv/) (`~/.local/bin/uv`), using its pip interface. `.venv` was created by `uv venv` with uv-managed CPython 3.12. Run uv commands from the project root: uv finds `.venv` automatically. The uv cache is shared by all projects on this machine (`cache-dir = "/media/abdelrahman/Data/.uv-cache"` in `~/.config/uv/uv.toml`). It's on the same drive as `.venv`, so installs are hardlinked from it. Don't add a project `uv.toml` that sets `link-mode = "copy"`. ComfyUI-Manager is also set to install with uv (`use_uv = True` in `user/__manager/config.ini`).

Use uv to install packages and the venv Python to run scripts and linting:

```bash
uv pip install <package>
uv pip list / uv pip show <package> / uv pip check
.venv/bin/python <script.py>
```

- Don't use `uv add` / `uv sync` / `uv lock`. This checkout tracks upstream ComfyUI's `pyproject.toml`, which has no dependency list, and syncing would remove packages installed by custom nodes.
- torch is the CUDA 12.8 build (`2.10.0+cu128`). To reinstall or change it, pass the index explicitly: `uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128`.

## Development Setup

```bash
# Python 3.10+ required
uv pip install -r requirements.txt
uv pip install -r manager_requirements.txt  # For manager features
```

## Running ComfyUI

```bash
python main.py
```

Common flags:
- `--enable-manager` - Enable ComfyUI-Manager for custom node management
- `--preview-method auto|taesd` - Configure preview quality
- `--front-end-version Comfy-Org/ComfyUI_frontend@latest` - Use latest frontend
- `--cpu` - Run without GPU support

## Testing

```bash
# Run all tests
pytest

# Run only unit tests (exclude inference/execution tests)
pytest -m "not inference and not execution"

# Run specific test markers
pytest -m inference    # Inference tests only
pytest -m execution    # Execution tests only
```

Test configuration in `pytest.ini`:
- Test paths: `tests/` and `tests-unit/`
- Markers: `inference`, `execution`
- Python path includes project root

## Code Style

- **Linting**: Ruff (configured in `pyproject.toml`)
  - Selected rules: N805, S307, S102, E, T, W, F
  - Ignored: E501 (line length), E722, E731, E712, E402, E741
- **Docstrings**: Not required (disabled in pylint config)
- Generated code in `**/generated/*.pyi` is excluded from linting

## Custom Nodes Development

**Location**: `custom_nodes/` directory. Each custom node lives in its own subdirectory.

### Discovery Mechanism

ComfyUI scans `custom_nodes/` at startup (`nodes.py:2207`):
1. Executes `prestartup_script.py` if present (runs before main imports)
2. Imports `__init__.py` or looks for `comfy_entrypoint()`
3. Registers all nodes in `NODE_CLASS_MAPPINGS` or nodes from `get_node_list()`
4. Supports `WEB_DIRECTORY` for frontend JS files

### Two Registration Patterns

#### Modern Pattern (Recommended) - `comfy_api` extension

```python
from comfy_api.latest import ComfyExtension, io

class MyNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MyNode",
            display_name="My Node",
            category="My Category",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("value", min=0, max=100, default=50, lazy=True),
            ],
            outputs=[io.Image.Output()],
        )
    
    @classmethod
    def execute(cls, image, value) -> io.NodeOutput:
        # image is torch.Tensor [B, H, W, C] float32 0-1
        return io.NodeOutput(image)

class MyExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [MyNode]

async def comfy_entrypoint() -> MyExtension:
    return MyExtension()

#### Legacy Pattern - `NODE_CLASS_MAPPINGS`

```python
class MyNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"images": ("IMAGE",)}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "my_category"
    
    def execute(self, images):
        return (images,)

NODE_CLASS_MAPPINGS = {
    "MyNode": MyNode,
}
```

### Key I/O Types

| Type | Description |
|------|-------------|
| `io.Image` | torch.Tensor `[B, H, W, C]` float32 0-1 |
| `io.Int` / `io.Float` | Numbers with min/max/step |
| `io.String` | Text (multiline option) |
| `io.Combo` | Dropdown selection |
| `io.Latent` / `io.Conditioning` / `io.CLIP` / `io.Vae` / `io.Model` | SD pipeline types |

## Custom Workflows

**Location**: `user/default/workflows/` directory. Our target for creating new workflows.

### Workflow JSON Format

Each workflow is a single-line JSON file with this top-level structure:

```json
{
  "id": "uuid-string",
  "revision": 0,
  "last_node_id": 179,
  "last_link_id": 358,
  "nodes": [...]
}
```

- `id` — UUID for the workflow session
- `revision` — revision counter
- `last_node_id` — highest node ID (increment when adding nodes)
- `last_link_id` — highest link ID (increment when adding connections)

### Node Object Structure

```json
{
  "id": 1,
  "type": "VAELoader",
  "pos": [x, y],
  "size": [width, height],
  "flags": {"collapsed": false},
  "order": 0,
  "mode": 0,
  "inputs": [...],
  "outputs": [...],
  "properties": {
    "Node name for S&R": "VAELoader",
    "models": [{"name": "model.safetensors", "directory": "vae"}]
  },
  "widgets_values": ["value1", 42],
  "color": "#af3535",
  "bgcolor": "#333333"
}
```

- `mode`: 0 = normal, 1 = always execute, 2 = bypassed, 4 = disabled
- `inputs`/`outputs` — each has `name`, `type`, `link` (link ID or null), `slot_index` (for outputs)
- `widgets_values` — positional array matching node widget order

### GetNode / SetNode Pattern

Used for wire management in complex workflows. `SetNode` assigns a named reference to an output, `GetNode` retrieves it elsewhere — avoids direct linking clutter.

### Existing Workflows

| File | Purpose |
|------|---------|
| `firered-image-edit-1.1-.json` | FireRed image editing pipeline |
| `firered-image-edit-1.1-MultiImageLoader.json` | Multi-image variant of FireRed edit |
| `Hunyuan3D-2.0-Multi-View-Workflow.json` | 3D generation from multi-view images |
| `hunyuan3d-dit-v2-0.json` | Hunyuan3D DiT v2.0 model workflow |
| `image_qwen_image_edit_2511.json` | Qwen image editing (Nov 2025 model) |
| `image_qwen_image.json` | Qwen image generation |
| `LTX I2V First Last Frame 2 Stage Workflow v6.json` | LTX image-to-video with frame control |
| `LTX-2.3_-_I2V_T2V_Basic_GGUF.json` | LTX 2.3 I2V/T2V basic GGUF workflow |
| `人物姿势迁移  Qwen_image_edit_2511.json` | Pose transfer with Qwen image edit |

## Important Gotchas

- Frontend bugs should be reported to [ComfyUI_frontend](https://github.com/Comfy-Org/ComfyUI_frontend), not this repo
- Custom nodes may break between releases - master branch can be very unstable
- Model files are large and stored in `models/` subdirectories by type
- The execution graph is incremental - unchanged nodes won't re-run between executions
