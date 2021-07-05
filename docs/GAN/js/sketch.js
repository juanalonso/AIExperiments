// const status = document.getElementById('status');
const generateButton = document.getElementById('generate');

async function init() {
    generateButton.textContent = 'Cargando el modelo...'
    let model = await tf.loadLayersModel('trained_model/model.json')
    generateButton.textContent = 'Generar más ejemplos'

    generateButton.addEventListener('click', async () => {
        predictAndDraw(model);
    });
      
    predictAndDraw(model)
    
}


function predictAndDraw(model) {
    generateButton.disabled = true;

    const seed = tf.randomNormal([4, 100])

    const prediction = model.predict(seed).mul(0.5).add(0.5)
    const predictions = tf.unstack(prediction)
    
    for (let f=4; f>=1; f--) {
        draw(predictions[f-1], f)
        // predictions[f-1].dispose()
    }
    
    function draw(tensor, index){
        const canvasElement = document.getElementById("model-output-"+String(index)) 
        tf.browser.toPixels(tensor, canvasElement).then(() => { 
            tensor.dispose()
        });
    }
    
    seed.dispose()
    prediction.dispose()

    console.log("Make sure we cleaned up 5 - ", tf.memory().numTensors)
    generateButton.disabled = false;
}

init()