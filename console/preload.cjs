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
    consoleVersion() {
      return ipcRenderer.invoke("console-version");
    },
    checkConsoleUpdate() {
      return ipcRenderer.invoke("check-console-update");
    },
    installConsoleUpdate() {
      return ipcRenderer.invoke("install-console-update");
    },
    subscribeUpdateProgress(callback) {
      const handler = (_event, progress) => callback(progress);
      ipcRenderer.on("console-update-progress", handler);
      return () =>
        ipcRenderer.removeListener("console-update-progress", handler);
    },
    openConsoleReleases() {
      return ipcRenderer.invoke("open-console-releases");
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
