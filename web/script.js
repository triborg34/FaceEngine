const videoFrame = document.getElementById('video-frame');
const dataList = document.getElementById('data-list');

// WebSocket connection to receive video frames
const ws = new WebSocket('ws://127.0.0.1:5000');

ws.onmessage = (event) => {
    // Update the video frame
    videoFrame.src = `data:image/jpeg;base64,${event.data}`;
};

ws.onerror = (error) => {
    console.error('WebSocket error:', error);
};

// Fetch real-time data from PocketBase
const pocketBaseUrl = 'http://127.0.0.1:8090';
const collectionName = 'known_face';

async function fetchRealTimeData() {
    try {
        const response = await fetch(`${pocketBaseUrl}/api/collections/${collectionName}/records?perPage=1000`);
        const data = await response.json();

        // Clear the current list
        dataList.innerHTML = '';

        // Populate the list with new data
        data.items.forEach(item => {
            const listItem = document.createElement('li');
            listItem.textContent = `${item.name}: ${item.embdanings.length} embeddings`;
            dataList.appendChild(listItem);
        });
    } catch (error) {
        console.error('Error fetching real-time data:', error);
    }
}

// Fetch data every 5 seconds
setInterval(fetchRealTimeData, 5000);

// Initial fetch
fetchRealTimeData();