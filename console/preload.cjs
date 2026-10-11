const { contextBridge, ipcRenderer } = require("electron");
contextBridge.exposeInMainWorld(
  "bello",
  Object.freeze({
    selectChoice(kind, name) {
      return ipcRenderer.invoke("select-choice", kind, name);
    },
    abortHold(held, action) {
      return ipcRenderer.invoke("abort-hold", held, action);
    },
    clearFault() {
      return ipcRenderer.invoke("clear-fault");
    },
    toggleFullscreen() {
      return ipcRenderer.invoke("toggle-fullscreen");
    },
    saveSettings(settings) {
      return ipcRenderer.invoke("save-settings", settings);
    },
    subscribe(callback) {
      const handler = (_event, snapshot) => callback(snapshot);
      ipcRenderer.on("snapshot", handler);
      return () => ipcRenderer.removeListener("snapshot", handler);
    },
    subscribeTabCycle(callback) {
      const handler = (_event, direction) => callback(direction);
      ipcRenderer.on("tab-cycle", handler);
      return () => ipcRenderer.removeListener("tab-cycle", handler);
    },
    modelUrl: "bello://assets/urdf/bello_mini.urdf",
  }),
);
