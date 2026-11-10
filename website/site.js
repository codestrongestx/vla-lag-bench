// Videos play only while on screen. The markup gives every clip native controls, so the page works without this
// script; the script hides them and plays clips as they scroll into view. The cover's mosaic has no autoplay attribute
// and no preload, so viewers who prefer reduced motion see its poster and never download it.
const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const mosaic = document.getElementById('mosaic');
const players = [...document.querySelectorAll('video[data-play]')];

function play(video) {
  const promise = video.play();
  // A browser may refuse autoplay (iOS Low Power Mode does); give clips their controls back.
  if (promise) promise.catch(() => { if (video !== mosaic) video.controls = true; });
}

if (!reduceMotion) {
  players.forEach((video) => { video.controls = false; });
  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting) play(entry.target);
        else entry.target.pause();
      }
    }, { threshold: 0.3 });
    [mosaic, ...players].forEach((video) => observer.observe(video));
  } else {
    [mosaic, ...players].forEach(play);
  }
}

// Each row restarts from its first frame.
document.querySelectorAll('[data-replay]').forEach((button) => {
  button.addEventListener('click', () => {
    const video = document.getElementById(button.dataset.replay);
    video.currentTime = 0;
    play(video);
  });
});

// On a phone, a row's tiles are small: tapping a row opens it full screen.
const phone = window.matchMedia('(max-width: 40rem)');
players.forEach((video) => {
  video.addEventListener('click', () => {
    if (!phone.matches || video.controls) return;
    if (video.requestFullscreen) video.requestFullscreen().catch(() => {});
    else if (video.webkitEnterFullscreen) video.webkitEnterFullscreen();
  });
});

document.querySelectorAll('[data-copy]').forEach((button) => {
  button.addEventListener('click', async () => {
    const text = document.getElementById(button.dataset.copy).textContent;
    try {
      await navigator.clipboard.writeText(text);
      button.textContent = 'Copied';
    } catch {
      button.textContent = 'Select and copy';
    }
    setTimeout(() => { button.textContent = 'Copy'; }, 1600);
  });
});
