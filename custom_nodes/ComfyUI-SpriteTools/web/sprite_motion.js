// Save as / Open buttons for sprite motion files.
//
// Save Sprite Motion sends the motion (JSON text) to the page after a run; "Save motion as..." asks where to
// write it: the browser's save window (Chrome / Edge: showSaveFilePicker, remembering the last folder), or a
// download in browsers without it. Load Sprite Motion's "Open motion file..." picks a .json anywhere, uploads it
// to input/sprite_motions/ and selects it.
const { app } = window.comfyAPI.app;
const { api } = window.comfyAPI.api;

const MOTION_DIR = "sprite_motions";
const PICKER_ID = "sprite-motion"; // the browser remembers the last folder per id
const JSON_TYPE = [{ description: "Sprite motion", accept: { "application/json": [".json"] } }];

function chain(obj, name, fn) {
    const orig = obj[name];
    obj[name] = function (...args) {
        const r = orig?.apply(this, args);
        fn.apply(this, args);
        return r;
    };
}

async function saveAs(text, suggestedName) {
    if (window.showSaveFilePicker) {
        let handle;
        try {
            handle = await window.showSaveFilePicker({ id: PICKER_ID, suggestedName, types: JSON_TYPE });
        } catch (e) {
            if (e.name === "AbortError") return null; // cancelled
            throw e;
        }
        const writable = await handle.createWritable();
        await writable.write(text);
        await writable.close();
        return handle.name;
    }
    // no save window API (Firefox, or the page isn't on localhost / https): a download, which asks where to save
    // when the browser is set to ask
    const url = URL.createObjectURL(new Blob([text], { type: "application/json" }));
    const a = Object.assign(document.createElement("a"), { href: url, download: suggestedName });
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
    return suggestedName + " (download)";
}

function pickFile() {
    return new Promise((resolve) => {
        const input = Object.assign(document.createElement("input"), { type: "file", accept: ".json,application/json" });
        input.onchange = () => resolve(input.files[0] ?? null);
        input.oncancel = () => resolve(null);
        input.click();
    });
}

async function openMotion(node) {
    let file;
    if (window.showOpenFilePicker) {
        try {
            [file] = await window.showOpenFilePicker({ id: PICKER_ID, types: JSON_TYPE });
            file = await file.getFile();
        } catch (e) {
            if (e.name === "AbortError") return;
            throw e;
        }
    } else {
        file = await pickFile();
    }
    if (!file) return;
    let data;
    try {
        data = JSON.parse(await file.text());
    } catch {
        data = null;
    }
    if (data?.sprite_motion !== 1) {
        alert(`${file.name} is not a sprite motion file (save one with Save Sprite Motion).`);
        return;
    }
    const body = new FormData();
    body.append("image", file, file.name);
    body.append("type", "input");
    body.append("subfolder", MOTION_DIR);
    const resp = await api.fetchApi("/upload/image", { method: "POST", body });
    if (resp.status !== 200) {
        alert(`Could not copy ${file.name} to ComfyUI: ${resp.status} ${resp.statusText}`);
        return;
    }
    const { name } = await resp.json(); // renamed "x (1).json" if a different file had the same name
    const combo = node.widgets.find((w) => w.name === "motion_file");
    if (!combo.options.values.includes(name)) {
        combo.options.values = [...combo.options.values, name].sort((a, b) =>
            a === "(none)" ? -1 : b === "(none)" ? 1 : a.localeCompare(b));
    }
    combo.value = name;
    combo.callback?.(name);
    node.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "SpriteTools.SpriteMotionFiles",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "SaveSpriteMotion") {
            chain(nodeType.prototype, "onNodeCreated", function () {
                this.spriteMotion = null;
                this.saveButton = this.addWidget("button", "Save motion as...", null, async () => {
                    if (!this.spriteMotion) {
                        alert("Run the workflow first; then this button saves its motion.");
                        return;
                    }
                    const name = this.widgets.find((w) => w.name === "name")?.value || "motion";
                    const suggested = this.spriteMotion.name || (name.endsWith(".json") ? name : name + ".json");
                    try {
                        const saved = await saveAs(this.spriteMotion.text, suggested);
                        if (saved) this.saveButton.label = `Saved ${saved}: save again...`;
                    } catch (e) {
                        alert(`Could not save the motion: ${e.message}`);
                    }
                    this.setDirtyCanvas(true, true);
                }, { serialize: false });
                this.saveButton.label = "Save motion as... (run first)";
            });
            chain(nodeType.prototype, "onExecuted", function (message) {
                const text = message?.sprite_motion?.[0];
                if (!text) return;
                this.spriteMotion = { text, name: message.sprite_motion_name?.[0] };
                const frames = message.sprite_motion_frames?.[0];
                this.saveButton.label = `Save motion as... (${frames} frames ready)`;
                this.setDirtyCanvas(true, true);
            });
        }
        if (nodeData.name === "LoadSpriteMotion") {
            chain(nodeType.prototype, "onNodeCreated", function () {
                this.addWidget("button", "Open motion file...", null, () => {
                    openMotion(this).catch((e) => alert(`Could not open the motion: ${e.message}`));
                }, { serialize: false });
            });
        }
    },
});
