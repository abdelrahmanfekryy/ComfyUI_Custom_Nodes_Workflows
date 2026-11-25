import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Upload button + drag-and-drop for the "Load Document (PDF / Image)" node.
// The core image upload widget only accepts images, so PDFs need their own picker.

const NODE_TYPE = "OCRLoadDocument";
const ACCEPT = ".pdf,application/pdf,image/png,image/jpeg,image/webp,image/bmp,image/tiff,image/gif";
const EXT_RE = /\.(pdf|png|jpe?g|webp|bmp|tiff?|gif)$/i;

async function uploadFile(file) {
  const body = new FormData();
  body.append("image", file);
  const resp = await api.fetchApi("/upload/image", { method: "POST", body });
  if (resp.status !== 200) {
    throw new Error(`${resp.status} ${resp.statusText}`);
  }
  const data = await resp.json();
  return data.subfolder ? `${data.subfolder}/${data.name}` : data.name;
}

function setDocument(node, path) {
  const widget = node.widgets?.find((w) => w.name === "document");
  if (!widget) return;
  const values = widget.options?.values;
  if (Array.isArray(values)) {
    // Drop the "(upload a PDF or image)" placeholder once a real file exists.
    const placeholder = values.findIndex((v) => v.startsWith("("));
    if (placeholder !== -1) values.splice(placeholder, 1);
    if (!values.includes(path)) {
      values.push(path);
      values.sort();
    }
  }
  widget.value = path;
  widget.callback?.(path);
  node.setDirtyCanvas?.(true, true);
}

async function handleFiles(node, files) {
  const file = [...(files || [])].find((f) => EXT_RE.test(f.name));
  if (!file) {
    if (files?.length) alert("Unsupported file. Use a PDF, PNG, JPEG, WebP, BMP, TIFF or GIF.");
    return false;
  }
  try {
    setDocument(node, await uploadFile(file));
  } catch (err) {
    console.error("[LightOnOCR] upload failed", err);
    alert(`Upload failed: ${err.message ?? err}`);
  }
  return true;
}

app.registerExtension({
  name: "LightOnOCR.DocumentUpload",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_TYPE) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      const node = this;

      const input = document.createElement("input");
      input.type = "file";
      input.accept = ACCEPT;
      input.style.display = "none";
      input.onchange = async () => {
        await handleFiles(node, input.files);
        input.value = "";
      };
      document.body.append(input);

      const button = this.addWidget("button", "choose file to upload", "upload", () => input.click());
      button.label = "choose file to upload (PDF / image)";
      button.serialize = false;
      button.options = { ...(button.options || {}), serialize: false };

      const onRemoved = this.onRemoved;
      this.onRemoved = function () {
        input.remove();
        return onRemoved?.apply(this, arguments);
      };
      return r;
    };

    nodeType.prototype.onDragOver = function (e) {
      return !!e?.dataTransfer?.types?.includes?.("Files");
    };
    nodeType.prototype.onDragDrop = function (e) {
      const files = e?.dataTransfer?.files;
      if (!files?.length) return false;
      handleFiles(this, files);
      return true;
    };
  },
});
