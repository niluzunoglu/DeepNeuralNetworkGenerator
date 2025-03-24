<script>
	let layerCount = 3;
	let layerNeurons = Array(layerCount).fill(10); 
	let activationFunction = 'relu';
	let learningRate = 0.001;
	let epochs = 10;
  
	let trainingResults = '';
	let predictionResults = '';
  
	$: {
	  layerNeurons = Array(layerCount).fill(layerNeurons[0] || 10);
	}
  
	async function trainModel() {
	  const response = await fetch('/api/train', {
		method: 'POST',
		headers: {
		  'Content-Type': 'application/json'
		},
		body: JSON.stringify({
		  layer_count: layerCount,
		  layer_neurons: layerNeurons,
		  activation_function: activationFunction,
		  learning_rate: learningRate,
		  epochs: epochs
		})
	  });
  
	  const data = await response.json();
	  trainingResults = JSON.stringify(data);
	}
  
	async function predict() {
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
  
	<div class="input-group">
	  <label for="layerCount">Katman Sayısı:</label>
	  <input type="number" id="layerCount" bind:value={layerCount}>
	</div>
  
	{#each Array(layerCount) as _, i}
	  <div class="input-group">
		<label for="neuronCount-{i}">Katman {i + 1} Nöron Sayısı:</label>
		<input type="number" id="neuronCount-{i}" bind:value={layerNeurons[i]}>
	  </div>
	{/each}
  
	<div class="input-group">
	  <label for="activationFunction">Aktivasyon Fonksiyonu:</label>
	  <select id="activationFunction" bind:value={activationFunction}>
		<option value="relu">ReLU</option>
		<option value="sigmoid">Sigmoid</option>
		<option value="tanh">Tanh</option>
	  </select>
	</div>
  
	<div class="input-group">
	  <label for="learningRate">Öğrenme Oranı:</label>
	  <input type="number" id="learningRate" bind:value={learningRate}>
	</div>
  
	<div class="input-group">
	  <label for="epochs">Epok Sayısı:</label>
	  <input type="number" id="epochs" bind:value={epochs}>
	</div>
  
	<div class="button-group">
	  <button on:click={trainModel}>Modeli Eğit</button>
	  <button on:click={predict}>Tahmin Yap</button>
	</div>
  
	<h2>Eğitim Sonuçları:</h2>
	<pre>{trainingResults}</pre>
  
	<h2>Tahmin Sonuçları:</h2>
	<pre>{predictionResults}</pre>
  
	<footer>
	  © 2025 Nilderland
	</footer>
  </main>
  
  <style>
	main {
	  font-family: sans-serif;
	  max-width: 800px;
	  margin: 0 auto;
	  padding: 20px;
	  background-color: #f9f9f9;
	  border-radius: 8px;
	  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
	}
  
	h1 {
	  text-align: center;
	  margin-bottom: 20px;
	  color: #333;
	}
  
	.input-group {
	  margin-bottom: 15px;
	}
  
	label {
	  display: block;
	  margin-bottom: 5px;
	  font-weight: bold;
	  color: #555;
	}
  
	input[type="number"],
	select {
	  width: 100%;
	  padding: 8px;
	  border: 1px solid #ccc;
	  border-radius: 4px;
	  box-sizing: border-box; /* padding'in genişliğe dahil olmasını sağlar */
	}
  
	.button-group {
	  display: flex;
	  justify-content: space-around;
	  margin-top: 20px;
	}
  
	button {
	  background-color: #4CAF50; /* Yeşil */
	  border: none;
	  color: white;
	  padding: 10px 20px;
	  text-align: center;
	  text-decoration: none;
	  display: inline-block;
	  font-size: 16px;
	  margin: 4px 2px;
	  cursor: pointer;
	  border-radius: 4px;
	}
  
	button:hover {
	  opacity: 0.8;
	}
  
	h2 {
	  margin-top: 30px;
	  color: #333;
	}
  
	pre {
	  background-color: #eee;
	  padding: 10px;
	  border-radius: 4px;
	  overflow-x: auto;
	}
  
	footer {
	  text-align: center;
	  margin-top: 30px;
	  color: #777;
	}
  </style>