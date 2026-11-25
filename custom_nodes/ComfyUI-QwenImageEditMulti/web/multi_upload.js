import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// "Load Images (Multi Upload)": an upload button that takes several files, drag-and-drop,
// and a thumbnail strip. The "images" text widget (one file per line) is the source of truth.

const NODE_TYPE = "LoadImagesMultiUpload";
const EXT_RE = /\.(png|jpe?g|webp|bmp|tiff?|gif)$/i;

const parseNames = (text) => (text || "").split("\n").map((s) => s.trim()).filter(Boolean);

async function uploadFile(file) {
  const body = new FormData();
  body.append("image", file);
  const resp = await api.fetchApi("/upload/image", { method: "POST", body });
  if (resp.status !== 200) throw new Error(`${resp.status} ${resp.statusText}`);
  const data = await resp.json();
  return data.subfolder ? `${data.subfolder}/${data.name}` : data.name;
}

function viewUrl(name) {
  const clean = name.replace(/\s*\[input\]$/, "");
  const slash = clean.lastIndexOf("/");
  const params = new URLSearchParams({
    filename: slash === -1 ? clean : clean.slice(slash + 1),
    subfolder: slash === -1 ? "" : clean.slice(0, slash),
    type: "input",
  });
  return api.apiURL(`/view?${params}`);
}

function namesWidget(node) {
  return node.widgets?.find((w) => w.name === "images");
}

function setNames(node, names) {
  const widget = namesWidget(node);
  if (!widget) return;
  const value = names.join("\n");
  widget.value = value;
  if (widget.inputEl) widget.inputEl.value = value;
  widget.callback?.(value);
  renderGallery(node);
  node.setDirtyCanvas?.(true, true);
}

function renderGallery(node) {
  const gallery = node.__multiUploadGallery;
  if (!gallery) return;
  const names = parseNames(namesWidget(node)?.value);
  gallery.replaceChildren();
  if (!names.length) {
    const hint = document.createElement("div");
    hint.textContent = "No images. Upload or drop files here.";
    hint.style.cssText = "opacity:.6;font-size:12px;padding:6px;";
    gallery.append(hint);
    return;
  }
  names.forEach((name, i) => {
    const cell = document.createElement("div");
    cell.title = `Picture ${i + 1}: ${name}`;
    cell.style.cssText =
      "position:relative;width:88px;height:88px;flex:0 0 auto;border-radius:6px;overflow:hidden;background:#111;";
    const img = document.createElement("img");
    img.src = viewUrl(name);
    img.style.cssText = "width:100%;height:100%;object-fit:contain;";
    const badge = document.createElement("span");
    badge.textContent = String(i + 1);
    badge.style.cssText =
      "position:absolute;left:3px;top:3px;font-size:11px;padding:0 5px;border-radius:8px;background:rgba(0,0,0,.7);color:#fff;";
    const remove = document.createElement("button");
    remove.textContent = "×";
    remove.title = "Remove";
    remove.style.cssText =
      "position:absolute;right:3px;top:3px;width:18px;height:18px;line-height:16px;padding:0;border:0;border-radius:9px;" +
      "background:rgba(0,0,0,.7);color:#fff;cursor:pointer;font-size:13px;";
    remove.onclick = (e) => {
      e.stopPropagation();
      const current = parseNames(namesWidget(node)?.value);
      current.splice(i, 1);
      setNames(node, current);
    };
    cell.append(img, badge, remove);
    gallery.append(cell);
  });
}

async function addFiles(node, files) {
  const images = [...(files || [])].filter((f) => EXT_RE.test(f.name));
  if (!images.length) {
    if (files?.length) alert("Unsupported file. Use PNG, JPEG, WebP, BMP, TIFF or GIF images.");
    return;
  }
  const names = parseNames(namesWidget(node)?.value);
  for (const file of images) {
    try {
      names.push(await uploadFile(file));
    } catch (err) {
      console.error("[QwenImageEditMulti] upload failed", err);
      alert(`Upload of ${file.name} failed: ${err.message ?? err}`);
    }
  }
  setNames(node, names);
}

app.registerExtension({
  name: "QwenImageEditMulti.MultiUpload",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_TYPE) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      const node = this;

      const input = document.createElement("input");
      input.type = "file";
      input.multiple = true;
      input.accept = "image/*";
      input.style.display = "none";
      input.onchange = async () => {
        await addFiles(node, input.files);
        input.value = "";
      };
      document.body.append(input);

      const upload = this.addWidget("button", "upload images", "upload", () => input.click());
      upload.label = "choose images to upload";
      upload.serialize = false;
      upload.options = { ...(upload.options || {}), serialize: false };

      const clear = this.addWidget("button", "clear images", "clear", () => setNames(node, []));
      clear.label = "clear all";
      clear.serialize = false;
      clear.options = { ...(clear.options || {}), serialize: false };

      const gallery = document.createElement("div");
      gallery.style.cssText = "display:flex;flex-wrap:wrap;gap:6px;padding:4px;overflow:auto;box-sizing:border-box;";
      this.__multiUploadGallery = gallery;
      this.addDOMWidget("gallery", "multi_upload_gallery", gallery, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => 100,
      });

      // Keep thumbnails in sync with manual edits of the text box.
      const widget = namesWidget(this);
      if (widget) {
        const cb = widget.callback;
        widget.callback = function () {
          const res = cb?.apply(this, arguments);
          renderGallery(node);
          return res;
        };
        widget.inputEl?.addEventListener("input", () => renderGallery(node));
      }
      renderGallery(this);

      const onRemoved = this.onRemoved;
      this.onRemoved = function () {
        input.remove();
        return onRemoved?.apply(this, arguments);
      };
      return r;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      renderGallery(this);
      return r;
    };

    nodeType.prototype.onDragOver = function (e) {
      return !!e?.dataTransfer?.types?.includes?.("Files");
    };
    nodeType.prototype.onDragDrop = function (e) {
      const files = e?.dataTransfer?.files;
      if (!files?.length) return false;
      addFiles(this, files);
      return true;
    };
  },
});
