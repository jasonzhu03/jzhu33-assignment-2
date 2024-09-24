document.getElementById('kmeans-form').addEventListener('submit', function(event) {
    event.preventDefault();

    let formData = new FormData(this);
    
    fetch('/kmeans', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert(data.error);
        } else {
            plotClusters(data.data, data.labels, data.centroids);
        }
    })
    .catch(error => console.error('Error:', error));
});

function plotClusters(data, labels, centroids) {
    let ctx = document.getElementById('cluster-chart').getContext('2d');
    
    let colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'brown', 'gray', 'pink', 'cyan'];
    let datasets = [];

    for (let i = 0; i < centroids.length; i++) {
        let clusterData = data.filter((_, idx) => labels[idx] === i);
        let clusterX = clusterData.map(point => point[0]);
        let clusterY = clusterData.map(point => point[1]);

        datasets.push({
            label: `Cluster ${i + 1}`,
            data: clusterData.map(point => ({x: point[0], y: point[1]})),
            backgroundColor: colors[i],
            pointRadius: 3
        });
    }

    // Plot centroids
    datasets.push({
        label: 'Centroids',
        data: centroids.map(point => ({x: point[0], y: point[1]})),
        backgroundColor: 'black',
        pointRadius: 5,
        pointStyle: 'star'
    });

    new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: datasets
        },
        options: {
            scales: {
                x: { beginAtZero: true },
                y: { beginAtZero: true }
            }
        }
    });
}