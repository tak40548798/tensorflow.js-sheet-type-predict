import { ControllerDataset } from './js/controller_dataset.js';
const { ipcRenderer } = require('electron')

window.addEventListener("load", function() {
    const imagefiles = document.getElementById("upload-image")
    const label = document.getElementById("label")
    const addMultiSmapleBtn = document.getElementById("addMultiSmaple")
    const addCaptureSampleBtm = document.getElementById("captureSample")
    const trainModelBtn = document.getElementById("trainModel")
    const labelDiv = document.getElementById("labelDiv")
    const loadModelBtn = document.getElementById("loadModel")
    const predictBtn = document.getElementById("predict")

    const SHEETCONTROLS = ['A', 'B', 'C'];
    const NUM_CLASSES = 3;
    const cropcanvas = document.getElementById("crop");

    const webcam = document.getElementById("webcam")
    let truncatedMobileNet;
    let model;
    let isPredicting = false;
    let videoWidth = 1920;
    let videoHeight = 1080;

    const controllerDataset = new ControllerDataset(NUM_CLASSES);

    // Set hyper params from UI values.
    const learningRateElement = document.getElementById('learningRate');
    const getLearningRate = () => +learningRateElement.value;

    const batchSizeFractionElement = document.getElementById('batchSizeFraction');
    const getBatchSizeFraction = () => +batchSizeFractionElement.value;

    const epochsElement = document.getElementById('epochs');
    const getEpochs = () => +epochsElement.value;

    const denseUnitsElement = document.getElementById('dense-units');
    const getDenseUnits = () => +denseUnitsElement.value;


    function rendererLabel() {
        for (let index = 0; index < SHEETCONTROLS.length; index++) {
            const value = SHEETCONTROLS[index];
            const option = document.createElement("option");
            option.value = index;
            option.innerHTML = `NUM-${index}-${value}`;
            label.appendChild(option)

            const span = document.createElement("span");
            span.innerHTML = `${index}-${value}`;
            span.classList.add(`id${index}`);
            span.style.marginRight = "10px"
            labelDiv.appendChild(span)
        }
    }

    function trainStatus(status) {
        document.getElementById("trainStatus").innerHTML = "TrainStatus->" + status
    }

    // Sets up and trains the classifier.
    async function train() {
        if (controllerDataset.xs == null) {
            throw new Error('Add some examples before training!');
        }

        // Creates a 2-layer fully connected model. By creating a separate model,
        // rather than adding layers to the mobilenet model, we "freeze" the weights
        // of the mobilenet model, and only train weights from the new model.
        model = tf.sequential({
            layers: [
                // Flattens the input to a vector so we can use it in a dense layer. While
                // technically a layer, this only performs a reshape (and has no training
                // parameters).
                tf.layers.flatten({ inputShape: truncatedMobileNet.outputs[0].shape.slice(1) }),
                // Layer 1.
                tf.layers.dense({
                    units: getDenseUnits(),
                    activation: 'relu',
                    kernelInitializer: 'varianceScaling',
                    useBias: true
                }),
                // Layer 2. The number of units of the last layer should correspond
                // to the number of classes we want to predict.
                tf.layers.dense({
                    units: NUM_CLASSES,
                    kernelInitializer: 'varianceScaling',
                    useBias: false,
                    activation: 'softmax'
                })
            ]
        });

        // Creates the optimizers which drives training of the model.
        const optimizer = tf.train.adam(getLearningRate());
        // We use categoricalCrossentropy which is the loss function we use for
        // categorical classification which measures the error between our predicted
        // probability distribution over classes (probability that an input is of each
        // class), versus the label (100% probability in the true class)>
        model.compile({ optimizer: optimizer, loss: 'categoricalCrossentropy' });

        // We parameterize batch size as a fraction of the entire dataset because the
        // number of examples that are collected depends on how many examples the user
        // collects. This allows us to have a flexible batch size.
        const batchSize =
            Math.floor(controllerDataset.xs.shape[0] * getBatchSizeFraction());
        if (!(batchSize > 0)) {
            throw new Error(
                `Batch size is 0 or NaN. Please choose a non-zero fraction.`);
        }

        console.log(controllerDataset)
            // Train the model! Model.fit() will shuffle xs & ys so we don't have to.
        model.fit(controllerDataset.xs, controllerDataset.ys, {
            batchSize,
            epochs: getEpochs(),
            callbacks: {
                onBatchEnd: async(batch, logs) => {
                    trainStatus(' Loss: ' + logs.loss.toFixed(5));
                },
                onTrainEnd: async(logs) => {
                    console.log(model);
                    predictBtn.disabled = "";
                    const saveResults = await model.save('downloads://MyModel-1');
                    console.log(saveResults);
                }
            }
        });
    }

    // Loads mobilenet and returns a model that returns the internal activation
    // we'll use as input to our classifier model.
    async function loadTruncatedMobileNet() {
        const mobilenet = await tf.loadLayersModel(
            'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');

        //https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json
        //https://storage.googleapis.com/tfjs-models/savedmodel/mobilenet_v2_1.0_224/weights_manifest.json

        // Return a model that outputs an internal activation.
        const layer = mobilenet.getLayer('conv_pw_13_relu');
        return tf.model({ inputs: mobilenet.inputs, outputs: layer.output });
    }

    function readFileToImageElement(file) {

        return new Promise((resolve, reject) => {
            let reader = new FileReader();
            reader.onload = function() {
                let image = document.createElement('img');
                image.src = this.result;
                image.onload = function() {
                    resolve(image)
                }
            }
            reader.readAsDataURL(file);
        });

    }

    function getTensorImgFromElement(element) {

        const imageTensor = tf.browser.fromPixels(element);
        const processedImgTensor = tf.tidy(() => imageTensor.expandDims(0).toFloat().div(127).sub(1));

        imageTensor.dispose();

        return processedImgTensor
    }

    async function addMultiSampleFromInputfile(files, label) {

        for (let index = 0; index < files.length; index++) {
            trainModelBtn.disabled = "disabled"

            const file = files[index];
            let image = await readFileToImageElement(file);
            let imageTensorNormalize = getTensorImgFromElement(image)

            controllerDataset.addExample(truncatedMobileNet.predict(imageTensorNormalize), label);

            imageTensorNormalize.dispose();
        }

        trainModelBtn.disabled = ""

        console.log(label, "ok")
        return "ok";
    }

    // handle media stream
    async function handleStream(inputElement, mediaStream) {
        if (inputElement.srcObject) {
            const stream = inputElement.srcObject;
            const tracks = stream.getTracks();
            tracks.forEach((track) => {
                track.stop();
            });
            inputElement.srcObject = null;
        }
        inputElement.srcObject = mediaStream;

        const mediaPlay = new Promise((resolve) => {
            inputElement.onloadedmetadata = () => {
                inputElement.play();
                resolve("media-played");
            };
        })

        return await mediaPlay
    }

    async function getDeviceList() {
        const devices = await navigator.mediaDevices.enumerateDevices();
        // frist access camera is reload

        const videoDevices = [];
        devices.forEach((device) => {
            if (device.kind === 'videoinput') {
                let fristDevice = device.label.split(' ')
                if (fristDevice[0] == "OKIOCAM") {
                    videoDevices.push(device);
                } else {
                    // fristDeviceId.push(device.deviceId)
                }
            }
        });
        return videoDevices;
    }

    // start predict loop
    function startPredict() {
        let basetime = Date.now();
        let fps = 1000 / 24;
        let { sx, sy, sw, sh, dx, dy, dw, dh, cropctx, cropcanvas } = getCanvasSetting();

        function loopHandler() {
            let now = Date.now();
            let check = now - basetime;
            if (check / fps >= 1) {
                basetime = now;
                cropctx.drawImage(webcam, sx, sy, sw, sh, dx, dy, dw, dh); // draw crop canvas
                execPredict();
            }
            requestAnimationFrame(loopHandler, fps);
        }

        loopHandler();
    }

    // crop 1:1 image in video center setting
    function getCanvasSetting(width, height) {

        // screenshot big square image from center
        let size = null;

        if (width && height) {
            if (width > height)
                size = height;
            if (width < height)
                size = width;
            if (width == height)
                size = width;
        }

        cropcanvas.width = size || 224;
        cropcanvas.height = size || 224;
        let cropctx = cropcanvas.getContext('2d');
        let sx = (videoWidth - videoHeight) / 2;
        if (videoWidth == videoHeight)
            sx = 0;

        if (cropcanvas.width !== cropcanvas.height)
            return console.error("crop size error!");

        return {
            sx: sx,
            sy: 0,
            sw: videoHeight,
            sh: videoHeight,
            dx: 0,
            dy: 0,
            dw: cropcanvas.width,
            dh: cropcanvas.height,
            cropctx: cropctx,
            cropcanvas: cropcanvas
        }
    }

    async function execPredict() {

        const imageTensorNormalize = getTensorImgFromElement(cropcanvas)

        const embeddings = truncatedMobileNet.predict(imageTensorNormalize);

        // Make a prediction through our newly-trained model using the embeddings
        // from mobilenet as input.
        const predictions = model.predict(embeddings);

        // Returns the index with the maximum probability. This number corresponds
        // to the class the model thinks is the most probable given the input.
        const predictedClass = predictions.as1D().argMax();
        const classId = (await predictedClass.data())[0];

        labelDiv.children.forEach(ele => {
            ele.style.color = "black"
        })
        document.getElementsByClassName(`id${classId}`)[0].style.color = "red"

        imageTensorNormalize.dispose();
    }

    async function getStream() {
        let devices = await getDeviceList();

        let constraints;

        if (devices.length) {
            constraints = {
                audio: false,
                video: {
                    width: {
                        exact: videoWidth
                    },
                    height: {
                        exact: videoHeight
                    },
                    deviceId: devices[0].deviceId
                }
            }
        } else {
            constraints = {
                audio: false,
                video: true
            }
        }

        devices.forEach((ele) => {
            let name = ele.label.split(' ')[0];
            if (name == 'MEIDCAM') {
                constraints.video.deviceId = ele.deviceId
            } else {
                constraints.video.deviceId = ele.deviceId
            }
            console.log(constraints)
        })

        const mediaStream = await navigator.mediaDevices.getUserMedia(constraints);

        truncatedMobileNet = await loadTruncatedMobileNet();

        await handleStream(webcam, mediaStream);

        // Warm up the model. This uploads weights to the GPU and compiles the WebGL
        // programs so the first time we collect data from the webcam it will be quick.
        let { sx, sy, sw, sh, dx, dy, dw, dh, cropctx } = getCanvasSetting();
        cropctx.drawImage(webcam, sx, sy, sw, sh, dx, dy, dw, dh)
        let image = tf.browser.fromPixels(cropcanvas);
        truncatedMobileNet.predict(image.expandDims(0));
        image.dispose();

        if (videoWidth != videoHeight) {
            document.getElementById("line1").style.display = "block"
            document.getElementById("line1").style.width = webcam.clientWidth + "px"
            document.getElementById("line1").style.top = webcam.clientHeight / 2 + "px";
            document.getElementById("line1").style.left = "10px";
            document.getElementById("line2").style.display = "block"
            document.getElementById("line2").style.height = webcam.clientHeight + "px"
            document.getElementById("line2").style.top = "0px";
            document.getElementById("line2").style.left = webcam.clientWidth / 2 + "px";

            // capture image or predict image area
            document.getElementById("webcamCrop").style.display = "block"
            document.getElementById("webcamCrop").style.width = webcam.clientHeight + "px"
            document.getElementById("webcamCrop").style.height = webcam.clientHeight + "px"
            document.getElementById("webcamCrop").style.top = `0px`;
            document.getElementById("webcamCrop").style.left = 　`${(webcam.clientWidth - webcam.clientHeight )/2}px`
        }

        return webcam
    }

    async function init() {
        try {
            document.getElementById('webcam').width = 224;
            document.getElementById('webcam').height = 224;
            webcam = await tf.data.webcam(document.getElementById('webcam'));
        } catch (e) {
            console.log(e);
        }
        truncatedMobileNet = await loadTruncatedMobileNet();

        // Warm up the model.This uploads weights to the GPU and compiles the WebGL
        // programs so the first time we collect data from the webcam it will be quick.
        const screenShot = await webcam.capture();
        console.log(screenShot)

        truncatedMobileNet.predict(screenShot.expandDims(0));
        screenShot.dispose();
    }

    // add multi sample from input file
    addMultiSmapleBtn.onclick = () => {
        addMultiSampleFromInputfile(imagefiles.files, parseInt(label.value))
    }

    // train
    trainModelBtn.onclick = () => {
        isPredicting = false
        train()
    }

    // predict 
    predictBtn.onclick = () => {
        isPredicting = true;

        if (isPredicting)
            cropcanvas.style.display = "inline"
        else
            cropcanvas.style.display = "none"

        startPredict();
    }

    // add single sample from video frame
    addCaptureSampleBtm.onclick = async() => {
        let { sx, sy, sw, sh, dx, dy, dw, dh, cropctx, cropcanvas } = getCanvasSetting();
        cropctx.drawImage(webcam, sx, sy, sw, sh, dx, dy, dw, dh); // draw crop canvas
        let imageTensorNormalize = getTensorImgFromElement(cropcanvas)

        controllerDataset.addExample(truncatedMobileNet.predict(imageTensorNormalize), parseInt(label.value));

        trainModelBtn.disabled = "";

        imageTensorNormalize.dispose();
    }

    const jsonUpload = document.getElementById("upload-json")
    const weightsUpload = document.getElementById("upload-weights")

    // load model
    loadModelBtn.onclick = async() => {
        if (jsonUpload.files[0] && weightsUpload.files[0]) {
            model = await tf.loadLayersModel(
                tf.io.browserFiles([jsonUpload.files[0], weightsUpload.files[0]]))

            addMultiSmapleBtn.disabled = "disabled";
            addCaptureSampleBtm.disabled = "disabled";
            trainModelBtn.disabled = "disabled"
            predictBtn.disabled = ""
        }
    }

    const clearBtn = document.getElementById("clear")

    clearBtn.onclick = () => {
        addMultiSmapleBtn.disabled = "";
        addCaptureSampleBtm.disabled = "";
        trainModelBtn.disabled = "disabled"
        predictBtn.disabled = "disabled";
        model = null;
    }

    // caputue image save to disk frome video frame
    const captureExampleBtn = document.getElementById("captureExample");

    captureExampleBtn.onclick = () => {

        let { sx, sy, sw, sh, dx, dy, dw, dh, cropctx, cropcanvas } = getCanvasSetting();
        cropctx.drawImage(webcam, sx, sy, sw, sh, dx, dy, dw, dh);
        let base64data = cropcanvas.toDataURL('image/jpeg', 1).split(';base64,')[1];
        ipcRendererSend("save-image", base64data, `${label.value}_${SHEETCONTROLS[label.value]}`);

        function drawRotated(ctx, canvas, degrees) {
            ctx.save();
            ctx.translate(canvas.width / 2, canvas.height / 2);
            ctx.rotate(degrees * Math.PI / 180);
            ctx.drawImage(canvas, -canvas.width / 2, -canvas.width / 2);
            ctx.restore();
            base64data = canvas.toDataURL('image/jpeg', 1).split(';base64,')[1];

            // 0_x_xx
            ipcRendererSend("save-image", base64data, `${label.value}_${SHEETCONTROLS[label.value]}`);
        }

        drawRotated(cropctx, cropcanvas, 90);
        drawRotated(cropctx, cropcanvas, 90);
        drawRotated(cropctx, cropcanvas, 90);
    }

    // electron.js api 
    function ipcRendererSend(event, data, arg) {
        ipcRenderer.send(event, data, arg)
    }

    getStream()

    rendererLabel()

    // init();
})