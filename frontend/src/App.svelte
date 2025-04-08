<script>
	import { SvelteComponent, tick } from 'svelte';

	let implementationMode = 'custom'; // 'custom' or 'library'
	let layerCount = 3;
	let layerNeurons = Array(layerCount).fill(10);
	let activationFunction = 'relu';
	let lossFunction = 'mse';
	let learningMethod = 'sgd';
	let learningRate = 0.001;
	let epochs = 10;

	let trainingResults = '';
	let predictionResults = '';
	let isLoadingTrain = false;
	let isLoadingPredict = false;
	let trainError = '';
	let predictError = '';

	$: if (layerNeurons.length !== layerCount) {
		const newLayerNeurons = [];
		const currentLength = layerNeurons.length;
		for (let i = 0; i < layerCount; i++) {
			newLayerNeurons[i] = i < currentLength ? layerNeurons[i] : 10;
		}
		layerNeurons = newLayerNeurons;
	}

	async function trainModel() {
		isLoadingTrain = true;
		trainingResults = '';
		trainError = '';
		try {
			const response = await fetch('/api/train', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					implementation_mode: implementationMode, // Added mode
					layer_count: Number(layerCount),
					layer_neurons: layerNeurons.map(Number),
					activation_function: activationFunction,
					loss_function: lossFunction,
					learning_method: learningMethod,
					learning_rate: Number(learningRate),
					epochs: Number(epochs)
				})
			});

			if (!response.ok) {
				let errorBody = `HTTP error! status: ${response.status}`;
				try {
					const errorData = await response.json();
					errorBody += ` - ${JSON.stringify(errorData)}`;
				} catch (e) { /* ignore */ }
				throw new Error(errorBody);
			}

			const data = await response.json();
			trainingResults = JSON.stringify(data, null, 2);
		} catch (error) {
			console.error('Training failed:', error);
			trainError = `Eğitim sırasında bir hata oluştu: ${error.message}`;
		} finally {
			isLoadingTrain = false;
		}
	}

	async function predict() {
		isLoadingPredict = true;
		predictionResults = '';
		predictError = '';
		const inputData = [0.1, 0.2, 0.3, 0.4, 0.5];
		try {
			const response = await fetch('/api/predict', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					// Assuming prediction API might also need mode or uses the trained model state
					implementation_mode: implementationMode,
					input_data: inputData
				})
			});

			if (!response.ok) {
				let errorBody = `HTTP error! status: ${response.status}`;
				try {
					const errorData = await response.json();
					errorBody += ` - ${JSON.stringify(errorData)}`;
				} catch (e) { /* ignore */ }
				throw new Error(errorBody);
			}

			const data = await response.json();
			predictionResults = JSON.stringify(data, null, 2);
		} catch (error) {
			console.error('Prediction failed:', error);
			predictError = `Tahmin sırasında bir hata oluştu: ${error.message}`;
			if (error.message.includes('404') || error.message.includes('500')) {
				predictError += " (Model eğitilmemiş veya API hatası olabilir)";
			}
		} finally {
			isLoadingPredict = false;
		}
	}
</script>

<main>
	<header>
		<h1>🧠 Derin Sinir Ağı Oluşturucu 🧠 </h1>
	</header>

    <section class="mode-selection card">
        <h2>Uygulama Modu Seçimi</h2>
        <div class="mode-options">
            <label class="mode-option">
                <input type="radio" bind:group={implementationMode} name="mode" value="custom">
                <div class="mode-content">
                    <span class="mode-title">Özel Uygulama</span>
                    <span class="mode-desc">(From Scratch / Temel Algoritmalar)</span>
                </div>
            </label>
            <label class="mode-option">
                <input type="radio" bind:group={implementationMode} name="mode" value="library">
                 <div class="mode-content">
                    <span class="mode-title">Kütüphane Kullan</span>
                    <span class="mode-desc">(TensorFlow / Keras vb.)</span>
                </div>
            </label>
        </div>
    </section>

	<section class="config-section card parameters-card">
		<h2>Parametreler</h2>

		<div class="input-grid">
			<div class="input-group">
				<label for="layerCount">Katman Sayısı:</label>
				<input type="number" id="layerCount" bind:value={layerCount} min="1" step="1">
			</div>
			<div class="input-group">
				<label for="learningRate">Öğrenme Oranı:</label>
				<input type="number" id="learningRate" bind:value={learningRate} step="0.0001" min="0.00001" max="1">
			</div>
			<div class="input-group">
				<label for="epochs">Epok Sayısı:</label>
				<input type="number" id="epochs" bind:value={epochs} min="1" step="1">
			</div>
		</div>

		<div class="input-row">
			<div class="input-group">
				<label for="activationFunction">Aktivasyon Fonksiyonu:</label>
				<select id="activationFunction" bind:value={activationFunction}>
					<option value="relu">ReLU</option>
					<option value="sigmoid">Sigmoid</option>
					<option value="tanh">Tanh</option>
                    <option value="linear">Linear</option>
                    <option value="softmax">Softmax</option>
				</select>
			</div>
			<div class="input-group">
				<label for="lossFunction">Loss Fonksiyonu:</label>
				<select id="lossFunction" bind:value={lossFunction}>
					<option value="mse">Mean Squared Error (MSE)</option>
					<option value="mae">Mean Absolute Error (MAE)</option>
                    <option value="categorical_crossentropy">Categorical Crossentropy</option>
                    <option value="binary_crossentropy">Binary Crossentropy</option>
				</select>
			</div>
            <div class="input-group">
				<label for="learningMethod">Öğrenme Şekli:</label>
				<select id="learningMethod" bind:value={learningMethod}>
					<option value="sgd">Stochastic Gradient Descent (SGD)</option>
					<option value="batch">Batch Gradient Descent</option>
					<option value="mini_batch">Mini-Batch Gradient Descent</option>
				</select>
			</div>
		</div>

        {#if layerCount > 0}
        <div class="neuron-section">
            <h3>Katman Nöronları</h3>
            <div class="neuron-container">
                {#each Array(layerCount) as _, i}
                    <div class="neuron-group">
                        <label for="neuronCount-{i}">Katman {i + 1}:</label>
                        <input
                            type="number"
                            id="neuronCount-{i}"
                            bind:value={layerNeurons[i]}
                            min="1"
                            step="1"
                            placeholder="Nöron"
                        >
                    </div>
                {/each}
            </div>
        </div>
        {/if}

	</section>

	<section class="action-section card actions-card">
		<h2>Eylemler</h2>
		<div class="button-group">
			<button on:click={trainModel} disabled={isLoadingTrain || isLoadingPredict}>
				{#if isLoadingTrain}
					Eğitiliyor... <span class="spinner"></span>
				{:else}
					<i class="fas fa-brain"></i> Modeli Eğit
				{/if}
			</button>
			<button on:click={predict} disabled={isLoadingTrain || isLoadingPredict}>
				{#if isLoadingPredict}
					Tahmin Ediliyor... <span class="spinner"></span>
				{:else}
					<i class="fas fa-search-location"></i> Tahmin Yap
				{/if}
			</button>
		</div>
	</section>

	{#if trainingResults || trainError}
		<section class="results-section card">
			<h2><i class="fas fa-chart-line"></i> Eğitim Sonuçları:</h2>
			{#if trainError}
				<p class="error-message">{trainError}</p>
			{/if}
			{#if trainingResults}
				<pre>{trainingResults}</pre>
			{/if}
		</section>
	{/if}

	{#if predictionResults || predictError}
		<section class="results-section card">
			<h2><i class="fas fa-bullseye"></i> Tahmin Sonuçları:</h2>
			{#if predictError}
				<p class="error-message">{predictError}</p>
			{/if}
			{#if predictionResults}
				<pre>{predictionResults}</pre>
			{/if}
		</section>
	{/if}

	<footer>
		© {new Date().getFullYear()} nilderland 
	</footer>
</main>

<!-- <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"> -->


<style>
	:root {
		--primary-color: #4a90e2;
		--primary-hover: #357ABD;
		--secondary-color: #f4f7fc;
		--card-bg: #ffffff;
		--text-color: #333;
		--label-color: #555;
		--border-color: #dfe7ef;
		--error-color: #d9534f;
		--success-color: #5cb85c;
		--spinner-color: var(--primary-hover);
        --card-bg-params: #eef6ff;
        --card-bg-actions: #f0fff4;
		--spacing-xs: 4px;
		--spacing-sm: 8px;
		--spacing-md: 16px;
		--spacing-lg: 24px;
		--spacing-xl: 32px;
		--border-radius: 8px;
		--font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol';
	}

	* { box-sizing: border-box; margin: 0; padding: 0; }
	body { background-color: var(--secondary-color); color: var(--text-color); font-family: var(--font-family); line-height: 1.6; }
	main { max-width: 900px; margin: var(--spacing-lg) auto; padding: var(--spacing-lg); display: flex; flex-direction: column; gap: var(--spacing-lg); }

	header { text-align: center; margin-bottom: var(--spacing-md); }
	header h1 { font-size: 2.2rem; margin-bottom: var(--spacing-sm); color: var(--primary-color); }

	.card {
		background-color: var(--card-bg);
		border-radius: var(--border-radius);
		padding: var(--spacing-lg);
		box-shadow: 0 5px 15px rgba(0, 0, 0, 0.07);
		border: 1px solid var(--border-color);
	}
    .parameters-card { background-color: var(--card-bg-params); }
    .actions-card { background-color: var(--card-bg-actions); }

    .mode-selection h2,
    .config-section h2,
    .action-section h2,
    .results-section h2 {
        font-size: 1.4rem;
        margin-bottom: var(--spacing-lg);
        color: var(--primary-color);
        border-bottom: 2px solid rgba(74, 144, 226, 0.3);
        padding-bottom: var(--spacing-sm);
        display: inline-block;
        font-weight: 600;
    }
    .results-section h2 i { margin-right: var(--spacing-sm); font-size: 1.2rem; }

    .mode-selection {
        background-color: var(--card-bg); /* Default card background */
    }
    .mode-options {
        display: flex;
        gap: var(--spacing-md);
        flex-wrap: wrap;
    }
    .mode-option {
        flex: 1 1 200px;
        display: flex;
        align-items: flex-start;
        padding: var(--spacing-md);
        border: 2px solid var(--border-color);
        border-radius: var(--border-radius);
        cursor: pointer;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
        background-color: #fff;
    }
     .mode-option input[type="radio"] {
        margin-top: 3px; /* Align radio button better */
        margin-right: var(--spacing-md);
        accent-color: var(--primary-color); /* Style the radio button */
        width: 1.2em;
        height: 1.2em;
    }
     .mode-option:hover {
        border-color: var(--primary-color);
    }
    .mode-option input[type="radio"]:checked + .mode-content {
        /* Optionally style the content when checked */
    }
     label.mode-option:has(input:checked) { /* Style the whole label when checked */
        border-color: var(--primary-color);
        box-shadow: 0 0 0 2px rgba(74, 144, 226, 0.2);
    }

    .mode-content {
        display: flex;
        flex-direction: column;
    }
    .mode-title {
        font-weight: 600;
        color: var(--text-color);
    }
    .mode-desc {
        font-size: 0.85rem;
        color: var(--label-color);
    }


    .input-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: var(--spacing-lg);
        margin-bottom: var(--spacing-lg);
    }
    .input-row {
        display: flex;
        flex-wrap: wrap;
        gap: var(--spacing-lg);
        margin-bottom: var(--spacing-lg);
    }
    .input-row > .input-group {
        flex: 1 1 180px;
    }

	.input-group {
		display: flex;
        flex-direction: column;
        gap: var(--spacing-sm);
	}

	label {
		font-weight: 600;
		color: var(--label-color);
		font-size: 0.95rem;
	}

	input[type="number"],
	select {
		width: 100%;
		padding: var(--spacing-sm) var(--spacing-md);
		border: 1px solid var(--border-color);
		border-radius: var(--border-radius);
		background-color: #fff;
		font-size: 1rem;
		transition: border-color 0.2s ease, box-shadow 0.2s ease;
        height: 42px;
        appearance: none;
	}
    select {
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16' fill='%23${'555'.toString(16)}'%3E%3Cpath fill-rule='evenodd' d='M4.22 6.22a.75.75 0 0 1 1.06 0L8 8.94l2.72-2.72a.75.75 0 1 1 1.06 1.06l-3.25 3.25a.75.75 0 0 1-1.06 0L4.22 7.28a.75.75 0 0 1 0-1.06z'/%3E%3C/svg%3E");
        background-repeat: no-repeat;
        background-position: right var(--spacing-md) center;
        background-size: 1em;
        padding-right: calc(var(--spacing-md) * 2.5);
    }


	input[type="number"]:focus,
	select:focus {
		outline: none;
		border-color: var(--primary-color);
		box-shadow: 0 0 0 3px rgba(74, 144, 226, 0.2);
	}
    input::placeholder { color: #aaa; font-style: italic; }

    .neuron-section {
        margin-top: var(--spacing-lg);
        padding-top: var(--spacing-md);
        border-top: 1px solid rgba(0,0,0,0.08);
    }
    .neuron-section h3 {
        font-size: 1.1rem;
        margin-bottom: var(--spacing-md);
        color: var(--label-color);
        font-weight: 600;
    }
	.neuron-container {
        display: flex;
        flex-wrap: wrap;
        gap: var(--spacing-md);
    }
	.neuron-group {
        /* Flex basis calculation: (100% / number of items) - gap adjustment */
        /* For 5 items with var(--spacing-md) gap: */
		flex: 1 1 calc(20% - var(--spacing-md) * 4 / 5); /* Adjust flex basis for 5 items */
		min-width: 110px; /* Adjust min-width if necessary */
        display: flex;
        flex-direction: column;
        gap: var(--spacing-xs);
	}
    .neuron-group label { font-size: 0.85rem; white-space: nowrap; }

	.button-group {
		display: flex; justify-content: center; gap: var(--spacing-lg);
		margin-top: var(--spacing-sm); flex-wrap: wrap;
	}
	button {
		background-color: var(--primary-color); border: none; color: white;
		padding: var(--spacing-md) var(--spacing-lg);
		font-size: 1rem; font-weight: 600; cursor: pointer;
		border-radius: var(--border-radius);
		transition: background-color 0.2s ease, transform 0.1s ease, box-shadow 0.2s ease;
		display: inline-flex; align-items: center; gap: var(--spacing-sm);
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        line-height: 1;
	}
    button i { font-size: 1.1em; }
	button:hover:not(:disabled) {
		background-color: var(--primary-hover);
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15);
		transform: translateY(-2px);
	}
	button:active:not(:disabled) { transform: translateY(0px); box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1); }
	button:disabled { background-color: #b0c4de; cursor: not-allowed; opacity: 0.8; }

	.results-section { background-color: lavenderblush; margin-top: var(--spacing-md); }
	pre {
		background-color: var(--card-bg);
        color: var(--text-color);
		padding: var(--spacing-md); border-radius: var(--border-radius);
		overflow-x: auto; font-family: 'Fira Code', 'Courier New', Courier, monospace;
		font-size: 0.9rem; border: 1px solid var(--border-color);
        white-space: pre-wrap; word-wrap: break-word;
	}
	.error-message {
		color: var(--error-color); background-color: rgba(217, 83, 79, 0.08);
		border: 1px solid rgba(217, 83, 79, 0.3);
		padding: var(--spacing-sm) var(--spacing-md); border-radius: var(--border-radius);
		margin-bottom: var(--spacing-md);
	}

	footer { text-align: center; margin-top: var(--spacing-lg); color: #888; font-size: 0.9rem; }

	.spinner { display: inline-block; width: 1em; height: 1em; border: 2px solid rgba(255, 255, 255, 0.3); border-radius: 50%; border-top-color: #fff; animation: spin 1s ease-in-out infinite; margin-left: var(--spacing-xs); }
	@keyframes spin { to { transform: rotate(360deg); } }

    @media (max-width: 768px) {
        .input-grid, .input-row { gap: var(--spacing-md); }
        .input-grid, .input-row > .input-group { min-width: 150px; }
        .neuron-group {
             flex-basis: calc(33.33% - var(--spacing-md) * 2 / 3); /* 3 items per row on medium screens */
             min-width: 100px;
        }
         .mode-options { flex-direction: column; } /* Stack mode options */
    }
    @media (max-width: 600px) {
        main { padding: var(--spacing-md); margin: var(--spacing-md) auto; }
        header h1 { font-size: 1.8rem; }
        .input-grid { grid-template-columns: 1fr; }
        .input-row { flex-direction: column; }
        .input-row > .input-group { flex-basis: auto; }
        .neuron-container { gap: var(--spacing-sm); }
        .neuron-group {
            flex-basis: calc(50% - var(--spacing-sm) / 2); /* 2 items per row */
            min-width: 100px;
        }
        .button-group { flex-direction: column; align-items: stretch; }
        button { padding: var(--spacing-sm) var(--spacing-md); }
    }

</style>