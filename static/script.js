let points = [];
let manual_centroids = [];
let kmeans_initialized = false;
let selecting_centroids = false;

document.getElementById('init_method').addEventListener('change', function () {
    const init_method = this.value;
    if (init_method === 'manual') {
        selecting_centroids = true;
        manual_centroids = [];
        alert('Please click on the graph to select centroids.');
    } else {
        selecting_centroids = false;
    }
});

// Capturing exact mouse click for centroids on the plot
function captureManualCentroid(event) {
    if (!selecting_centroids) return;

    const plotDiv = document.getElementById('plot');
    const rect = plotDiv.getBoundingClientRect();
    const xPos = event.clientX - rect.left; // X coordinate relative to the plot
    const yPos = event.clientY - rect.top;  // Y coordinate relative to the plot

    const xAxis = plotDiv._fullLayout.xaxis;
    const yAxis = plotDiv._fullLayout.yaxis;

    // Translate pixel coordinates to data coordinates
    const xValue = xAxis.p2c(xPos);
    const yValue = yAxis.p2c(yPos);

    console.log(`Manual centroid added at: (${xValue}, ${yValue})`);

    manual_centroids.push([xValue, yValue]);
    plotResults(points, manual_centroids, []);
}

// Plot the results using Plotly
function plotResults(points, centroids, labels) {
    const tracePoints = {
        x: points.map(p => p[0]),
        y: points.map(p => p[1]),
        mode: 'markers',
        marker: {
            size: 8,
            color: labels,
            colorscale: 'Viridis'
        },
        name: 'Data Points',
        type: 'scatter'
    };

    const traceCentroids = centroids.length ? {
        x: centroids.map(c => c[0]),
        y: centroids.map(c => c[1]),
        mode: 'markers',
        marker: {
            size: 16,
            color: 'red',
            symbol: 'x'
        },
        name: 'Centroids',
        type: 'scatter'
    } : null;

    const data = traceCentroids ? [tracePoints, traceCentroids] : [tracePoints];

    const layout = {
        xaxis: { range: [-10, 10], scaleanchor: "y", scaleratio: 1 },
        yaxis: { range: [-10, 10] },
        title: 'KMeans Clustering Data',
        width: 800,  // Make the plot larger
        height: 800  // Make the plot larger
    };

    Plotly.newPlot('plot', data, layout);
}

// Event listener for mouse clicks on the plot
document.getElementById('plot').addEventListener('click', captureManualCentroid);

// KMeans and other functions (unchanged)
async function initializeKMeans() {
    const n_clusters = document.getElementById('n_clusters').value;
    const init_method = document.getElementById('init_method').value;

    if (init_method === 'manual' && manual_centroids.length === 0) {
        alert('Please select centroids manually by clicking on the graph.');
        return;
    }

    const response = await fetch('/run_kmeans', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ points, n_clusters, init_method, manual_centroids })
    });

    const data = await response.json();
    const { centroids, labels } = data;

    plotResults(points, centroids, labels);
    kmeans_initialized = true;
}

async function stepKMeans() {
    if (!kmeans_initialized) {
        await initializeKMeans();
    }

    const response = await fetch('/step_kmeans', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
    });

    const data = await response.json();
    const { centroids, labels, converged } = data;

    if (converged) {
        alert('KMeans has converged!');
    }

    plotResults(points, centroids, labels);
}

async function runToConvergence() {
    if (!kmeans_initialized) {
        await initializeKMeans();
    }

    const response = await fetch('/run_to_convergence', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
    });

    const data = await response.json();
    const { centroids, labels } = data;

    plotResults(points, centroids, labels);
    alert('KMeans has converged!');
}

async function generateNewDataset() {
    points = [...Array(300)].map(() => [Math.random() * 20 - 10, Math.random() * 20 - 10]);
    manual_centroids = [];
    plotResults(points, [], []);
    kmeans_initialized = false;
}

async function resetAlgorithm() {
    await fetch('/reset', {
        method: 'POST',
    });

    plotResults(points, [], []);
    kmeans_initialized = false;
    manual_centroids = [];
}

generateNewDataset();