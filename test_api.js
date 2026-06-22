const fs = require('fs');
const path = require('path');

async function test() {
    const formData = new FormData();
    const imagePath = "D:\\New folder\\Lab-grown samples\\Lab-grown samples\\Nurtient Plates\\WIN_20250905_11_43_22_Pro.jpg";
    
    // Check if test image exists, if not, create a dummy one or fail
    if (!fs.existsSync(imagePath)) {
        console.error('Test image not found at:', imagePath);
        return;
    }

    const blob = new Blob([fs.readFileSync(imagePath)]);
    formData.append('image', blob, 'test_image.jpg');
    formData.append('threshold_value', '15');

    try {
        const response = await fetch('http://localhost:3000/detect', {
            method: 'POST',
            body: formData
        });
        const result = await response.json();
        console.log('Result:', JSON.stringify(result, null, 2));
    } catch (error) {
        console.error('Test failed:', error);
    }
}

// Ensure fetch is available (Node 18+) or use node-fetch
if (!global.fetch) {
    console.log("Fetch not available in this Node version.");
} else {
    test();
}
