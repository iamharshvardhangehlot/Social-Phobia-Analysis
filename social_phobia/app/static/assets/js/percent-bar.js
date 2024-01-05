// Get elements
let number = document.getElementById('per-number');
let circle = document.querySelector('per-circle');
let per_counter = 0;
let animationDuration = 2000;

let intervalId = setInterval(() => {
    if (per_counter == 50) {
        clearInterval(intervalId);
    } else {
        per_counter += 1;
        number.innerHTML = `${per_counter}%`;
        
        // Calculate the animation duration based on the per_counter
        animationDuration = (per_counter / 70) * 2000; 
        
        // Set the --percentage custom property for the animation
        circle.style.setProperty('--percentage', per_counter);
        
        // Set the animation duration for the circle
        circle.style.animationDuration = `${animationDuration}ms`;
    }
}, 30);
