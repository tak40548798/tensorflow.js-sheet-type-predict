// main.js

// Modules to control application life and create native browser window
const { app, ipcMain, BrowserWindow } = require('electron')
const fs = require('fs');
const path = require('path');

const defaultPATH = __dirname;

function createWindow() {
    // Create the browser window.
    const mainWindow = new BrowserWindow({
        width: 1600,
        height: 1200,
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false
        }
    })

    // and load the index.html of the app.
    mainWindow.loadFile('main.html')

    // Open the DevTools.
    mainWindow.webContents.openDevTools()
}

function ipcMainOnLoad() {

    const imageFolderName = "image_" + Date.now();
    const imageFolderPath = path.join(defaultPATH, 'TrainImage', imageFolderName)

    ipcMain.on('save-image', (event, base64Data, label) => {

        let savePath = path.join(imageFolderPath, label.split('_')[1])

        console.log(label.split('_'))
        fs.mkdirSync(savePath, { recursive: true });

        savePath = path.join(savePath, `IMAGE_${Date.now()}.jpeg`)

        fs.writeFile(savePath, base64Data, 'base64', function(err) {
            console.log(err);
        });

    })

}

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
// 部分 API 在 ready 事件触发后才能使用。
app.whenReady().then(() => {
    createWindow()
    ipcMainOnLoad()
    app.on('activate', function() {
        // On macOS it's common to re-create a window in the app when the
        // dock icon is clicked and there are no other windows open.
        if (BrowserWindow.getAllWindows().length === 0) createWindow()
    })
})

// Quit when all windows are closed, except on macOS. There, it's common
// for applications and their menu bar to stay active until the user quits
// explicitly with Cmd + Q.
app.on('window-all-closed', function() {
    if (process.platform !== 'darwin') app.quit()
})

// In this file you can include the rest of your app's specific main process
// code. 也可以拆分成几个文件，然后用 require 导入。