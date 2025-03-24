<script>
	let layerCount = 3;
	let neuronCount = 10;
	let activationFunction = 'relu';
	let learningRate = 0.001;
	let epochs = 10;
  
	let trainingResults = '';
	let predictionResults = '';
  
	async function trainModel() {
	  const response = await fetch('/api/train', {
		method: 'POST',
		headers: {
		  'Content-Type': 'application/json'
		},
		body: JSON.stringify({
		  layer_count: layerCount,
		  neuron_count: neuronCount,
		  activation_function: activationFunction,
		  learning_rate: learningRate,
		  epochs: epochs
		})
	  });
  
	  const data = await response.json();
	  trainingResults = JSON.stringify(data);
	}
  
	async function predict() {
	  // Örnek girdi verisi
	  const inputData = [0.1, 0.2, 0.3, 0.4, 0.5];
  
	  const response = await fetch('/api/predict', {
		method: 'POST',
		headers: {
		  'Content-Type': 'application/json'
		},
		body: JSON.stringify({
		  input_data: inputData
		})
	  });
  
	  const data = await response.json();
	  predictionResults = JSON.stringify(data);
	}
  </script>
  
  <main>
	<h1>Derin Sinir Ağı Oluşturucu</h1>
  
	<div>
	  <label for="layerCount">Katman Sayısı:</label>
	  <input type="number" id="layerCount" bind:value={layerCount}>
	</div>
  
	<div>
	  <label for="neuronCount">Nöron Sayısı:</label>
	  <input type="number" id="neuronCount" bind:value={neuronCount}>
	</div>
  
	<div>
	  <label for="activationFunction">Aktivasyon Fonksiyonu:</label>
	  <select id="activationFunction" bind:value={activationFunction}>
		<option value="relu">ReLU</option>
		<option value="sigmoid">Sigmoid</option>
		<option value="tanh">Tanh</option>
	  </select>
	</div>
  
	<div>
	  <label for="learningRate">Öğrenme Oranı:</label>
	  <input type="number" id="learningRate" bind:value={learningRate}>
	</div>
  
	<div>
	  <label for="epochs">Epok Sayısı:</label>
	  <input type="number" id="epochs" bind:value={epochs}>
	</div>
  
	<button on:click={trainModel}>Modeli Eğit</button>
	<button on:click={predict}>Tahmin Yap</button>
  
	<h2>Eğitim Sonuçları:</h2>
	<pre>{trainingResults}</pre>
  
	<h2>Tahmin Sonuçları:</h2>
	<pre>{predictionResults}</pre>
  </main>
  
  <style>
	main {
	  font-family: sans-serif;
	  padding: 20px;
	}
  
	input, select {
	  margin-bottom: 10px;
	}
  
	pre {
	  background-color: #f0f0f0;
	  padding: 10px;
	  overflow-x: auto;
	}
  </style>