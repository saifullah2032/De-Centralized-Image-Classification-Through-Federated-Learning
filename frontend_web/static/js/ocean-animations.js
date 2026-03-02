/**
 * OCEAN BACKGROUNDS ANIMATION MODULE
 * Handles dynamic ocean effects for home page and subtle animations for other pages
 * Comic book ocean aesthetic with animated fish, coral, and bubbles
 */

class OceanAnimationManager {
  constructor() {
    this.pageType = this.detectPageType();
    this.animationFrameId = null;
    this.init();
  }

  /**
   * Detect page type to determine animation style
   * Returns: 'home' | 'secondary'
   */
  detectPageType() {
    const body = document.body;
    
    if (body.classList.contains('page-home')) return 'home';
    if (body.classList.contains('page-secondary')) return 'secondary';
    
    // Default: secondary (subtle) for all other pages
    return 'secondary';
  }

  /**
   * Initialize animations based on page type
   */
  init() {
    if (this.pageType === 'home') {
      this.initHomePageAnimation();
    } else {
      this.initSecondaryPageAnimation();
    }
  }

  /**
   * Initialize full ocean animation for home page
   */
  initHomePageAnimation() {
    // Create ocean background if not already present
    if (!document.querySelector('.ocean-background')) {
      this.createOceanBackground();
    }

    // Create animated container for fish and coral
    if (!document.querySelector('.ocean-animated-container')) {
      this.createAnimatedContainer();
    }

    // Load SVG elements dynamically
    this.loadSVGElements();

    // Add page class to body
    document.body.classList.add('page-home');
  }

  /**
   * Initialize subtle animation for secondary pages
   */
  initSecondaryPageAnimation() {
    // Create subtle background if not already present
    if (!document.querySelector('.subtle-ocean-bg')) {
      this.createSubtleBackground();
    }

    // Create subtle bubbles container
    if (!document.querySelector('.subtle-bubbles')) {
      this.createSubtleBubbles();
    }

    // Add page class to body
    document.body.classList.add('page-secondary');
  }

  /**
   * Create ocean background div
   */
  createOceanBackground() {
    const bg = document.createElement('div');
    bg.className = 'ocean-background';
    document.body.insertBefore(bg, document.body.firstChild);
  }

  /**
   * Create animated container for SVG elements
   */
  createAnimatedContainer() {
    const container = document.createElement('div');
    container.className = 'ocean-animated-container';

    // Create coral containers
    const coralContainer = document.createElement('div');
    coralContainer.className = 'coral-container';
    for (let i = 0; i < 6; i++) {
      const coral = document.createElement('div');
      coral.className = 'coral';
      coral.innerHTML = this.getCoralSVG(i);
      coralContainer.appendChild(coral);
    }

    // Create fish containers
    const fishContainer = document.createElement('div');
    fishContainer.className = 'fish-container';
    for (let i = 0; i < 3; i++) {
      const fish = document.createElement('div');
      fish.className = 'fish';
      fish.innerHTML = this.getFishSVG(i);
      fishContainer.appendChild(fish);
    }

    container.appendChild(coralContainer);
    container.appendChild(fishContainer);
    document.body.appendChild(container);
  }

  /**
   * Get coral SVG (simplified inline version) - Comic style coral branches
   */
  getCoralSVG(index) {
    const coralVariants = [
      // Coral 1 - Branching coral with polyps
      `<svg viewBox="0 0 80 120" xmlns="http://www.w3.org/2000/svg">
        <!-- Main stem -->
        <path d="M 40 120 Q 40 90 35 60 Q 33 40 38 20" stroke="#000000" stroke-width="3" fill="none" stroke-linecap="round"/>
        <!-- Left branches -->
        <path d="M 32 90 Q 15 85 8 70" stroke="#000000" stroke-width="2.5" fill="none" stroke-linecap="round"/>
        <circle cx="8" cy="68" r="5" fill="#ff6b6b" stroke="#000000" stroke-width="1"/>
        <path d="M 30 65 Q 12 60 5 48" stroke="#000000" stroke-width="2" fill="none" stroke-linecap="round"/>
        <circle cx="4" cy="46" r="4" fill="#ff4757" stroke="#000000" stroke-width="1"/>
        <!-- Right branches -->
        <path d="M 45 90 Q 65 85 72 70" stroke="#000000" stroke-width="2.5" fill="none" stroke-linecap="round"/>
        <circle cx="72" cy="68" r="5" fill="#ff6b6b" stroke="#000000" stroke-width="1"/>
        <path d="M 44 65 Q 62 60 75 48" stroke="#000000" stroke-width="2" fill="none" stroke-linecap="round"/>
        <circle cx="75" cy="46" r="4" fill="#ff4757" stroke="#000000" stroke-width="1"/>
        <!-- Root/base -->
        <ellipse cx="40" cy="118" rx="8" ry="6" fill="#ff6b6b" stroke="#000000" stroke-width="2"/>
      </svg>`,
      // Coral 2 - Tall branching coral
      `<svg viewBox="0 0 80 120" xmlns="http://www.w3.org/2000/svg">
        <!-- Main stem -->
        <path d="M 40 120 Q 42 95 38 65 Q 36 45 42 25" stroke="#000000" stroke-width="3" fill="none" stroke-linecap="round"/>
        <!-- Left branches with polyps -->
        <path d="M 36 95 Q 18 90 10 75" stroke="#000000" stroke-width="2.5" fill="none" stroke-linecap="round"/>
        <circle cx="10" cy="75" r="4" fill="#ff8787" stroke="#000000" stroke-width="1"/>
        <path d="M 34 70 Q 14 65 6 52" stroke="#000000" stroke-width="2.2" fill="none" stroke-linecap="round"/>
        <circle cx="6" cy="52" r="3.5" fill="#ff5e78" stroke="#000000" stroke-width="1"/>
        <!-- Right branches -->
        <path d="M 44 95 Q 62 90 70 75" stroke="#000000" stroke-width="2.5" fill="none" stroke-linecap="round"/>
        <circle cx="70" cy="75" r="4" fill="#ff8787" stroke="#000000" stroke-width="1"/>
        <path d="M 46 70 Q 66 65 74 52" stroke="#000000" stroke-width="2.2" fill="none" stroke-linecap="round"/>
        <circle cx="74" cy="52" r="3.5" fill="#ff5e78" stroke="#000000" stroke-width="1"/>
        <!-- Base -->
        <ellipse cx="40" cy="118" rx="8" ry="6" fill="#ff8787" stroke="#000000" stroke-width="2"/>
      </svg>`,
      // Coral 3 - Bushy coral with multiple branches
      `<svg viewBox="0 0 80 120" xmlns="http://www.w3.org/2000/svg">
        <!-- Main stem -->
        <path d="M 40 120 Q 38 85 42 50 Q 44 25 38 10" stroke="#000000" stroke-width="3" fill="none" stroke-linecap="round"/>
        <!-- Multiple branching points -->
        <path d="M 38 100 Q 20 95 12 85" stroke="#000000" stroke-width="2.2" fill="none" stroke-linecap="round"/>
        <circle cx="12" cy="85" r="4" fill="#ff6b6b" stroke="#000000" stroke-width="1"/>
        <path d="M 42 100 Q 60 95 68 85" stroke="#000000" stroke-width="2.2" fill="none" stroke-linecap="round"/>
        <circle cx="68" cy="85" r="4" fill="#ff6b6b" stroke="#000000" stroke-width="1"/>
        <path d="M 35 70 Q 15 65 8 55" stroke="#000000" stroke-width="2" fill="none" stroke-linecap="round"/>
        <circle cx="8" cy="55" r="3.5" fill="#ff4757" stroke="#000000" stroke-width="1"/>
        <path d="M 45 70 Q 65 65 72 55" stroke="#000000" stroke-width="2" fill="none" stroke-linecap="round"/>
        <circle cx="72" cy="55" r="3.5" fill="#ff4757" stroke="#000000" stroke-width="1"/>
        <!-- Base -->
        <ellipse cx="40" cy="118" rx="7" ry="5" fill="#ff6b6b" stroke="#000000" stroke-width="2"/>
      </svg>`,
      // Coral 4 - Curved elegant coral
      `<svg viewBox="0 0 80 120" xmlns="http://www.w3.org/2000/svg">
        <!-- Curved main stem -->
        <path d="M 40 120 Q 35 95 40 65 Q 42 40 40 15" stroke="#000000" stroke-width="3" fill="none" stroke-linecap="round"/>
        <!-- Left sweeping branches -->
        <path d="M 38 90 Q 10 88 5 75" stroke="#000000" stroke-width="2.5" fill="none" stroke-linecap="round"/>
        <circle cx="5" cy="75" r="4" fill="#ff8787" stroke="#000000" stroke-width="1"/>
        <path d="M 40 55 Q 18 50 10 38" stroke="#000000" stroke-width="2.2" fill="none" stroke-linecap="round"/>
        <circle cx="10" cy="38" r="3.5" fill="#ff5e78" stroke="#000000" stroke-width="1"/>
        <!-- Right sweeping branches -->
        <path d="M 42 90 Q 70 88 75 75" stroke="#000000" stroke-width="2.5" fill="none" stroke-linecap="round"/>
        <circle cx="75" cy="75" r="4" fill="#ff8787" stroke="#000000" stroke-width="1"/>
        <path d="M 40 55 Q 62 50 70 38" stroke="#000000" stroke-width="2.2" fill="none" stroke-linecap="round"/>
        <circle cx="70" cy="38" r="3.5" fill="#ff5e78" stroke="#000000" stroke-width="1"/>
        <!-- Base -->
        <ellipse cx="40" cy="118" rx="7" ry="5" fill="#ff8787" stroke="#000000" stroke-width="2"/>
      </svg>`
    ];
    return coralVariants[index % coralVariants.length];
  }

  /**
    * Get fish SVG (simplified inline version)
    */
    getFishSVG(index) {
      const fishVariants = [
        // Fish 1 - Pufferfish style (cute and round) - MIRRORED (eye on right, tail on left)
        `<svg viewBox="0 0 60 50" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <linearGradient id="fishGrad1" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" style="stop-color:#72edf1;stop-opacity:1" />
              <stop offset="100%" style="stop-color:#5fd4d8;stop-opacity:1" />
            </linearGradient>
          </defs>
          <!-- Body -->
          <ellipse cx="30" cy="25" rx="22" ry="18" fill="url(#fishGrad1)" stroke="#000000" stroke-width="2"/>
          <!-- Spikes -->
          <circle cx="48" cy="15" r="3" fill="#72edf1" stroke="#000000" stroke-width="1.5"/>
          <circle cx="12" cy="15" r="3" fill="#72edf1" stroke="#000000" stroke-width="1.5"/>
          <circle cx="45" cy="35" r="3" fill="#72edf1" stroke="#000000" stroke-width="1.5"/>
          <circle cx="15" cy="35" r="3" fill="#72edf1" stroke="#000000" stroke-width="1.5"/>
          <!-- Eye (on right side now) -->
          <circle cx="40" cy="20" r="3" fill="#000000"/>
          <circle cx="39" cy="19" r="1.5" fill="#ffffff"/>
          <!-- Mouth -->
          <path d="M 30 32 Q 28 35 30 37" stroke="#000000" stroke-width="2" fill="none" stroke-linecap="round"/>
          <!-- Tail (on left side now) -->
          <path d="M 8 25 L -2 20 L 0 25 L -2 30 Z" fill="#5fd4d8" stroke="#000000" stroke-width="1.5"/>
        </svg>`,
        // Fish 2 - Angelfish style (tall and elegant) - MIRRORED (eye on right, tail on left)
        `<svg viewBox="0 0 50 70" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <linearGradient id="fishGrad2" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" style="stop-color:#72edf1;stop-opacity:1" />
              <stop offset="100%" style="stop-color:#5fd4d8;stop-opacity:1" />
            </linearGradient>
          </defs>
          <!-- Body -->
          <ellipse cx="30" cy="35" rx="15" ry="25" fill="url(#fishGrad2)" stroke="#000000" stroke-width="2"/>
          <!-- Top fin -->
          <path d="M 35 10 Q 32 5 30 10" fill="#5fd4d8" stroke="#000000" stroke-width="1.5"/>
          <!-- Bottom fin -->
          <path d="M 35 60 Q 32 65 30 60" fill="#5fd4d8" stroke="#000000" stroke-width="1.5"/>
          <!-- Eye (on right side) -->
          <circle cx="40" cy="30" r="2.5" fill="#000000"/>
          <circle cx="39" cy="29" r="1" fill="#ffffff"/>
          <!-- Gill -->
          <path d="M 38 35 Q 42 35 44 33" stroke="#000000" stroke-width="1.5" fill="none"/>
          <!-- Tail (on left side) -->
          <path d="M 15 30 L 2 20 L 4 35 L 2 50 Z" fill="#5fd4d8" stroke="#000000" stroke-width="1.5"/>
        </svg>`,
        // Fish 3 - Tropical fish (striped) - MIRRORED (eye on right, tail on left)
        `<svg viewBox="0 0 70 45" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <linearGradient id="fishGrad3" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" style="stop-color:#72edf1;stop-opacity:1" />
              <stop offset="50%" style="stop-color:#ecf0f1;stop-opacity:1" />
              <stop offset="100%" style="stop-color:#5fd4d8;stop-opacity:1" />
            </linearGradient>
          </defs>
          <!-- Body -->
          <ellipse cx="40" cy="22" rx="20" ry="15" fill="url(#fishGrad3)" stroke="#000000" stroke-width="2"/>
          <!-- Stripe 1 -->
          <path d="M 50 10 Q 45 22 50 35" stroke="#000000" stroke-width="1.5" fill="none" opacity="0.5"/>
          <!-- Stripe 2 -->
          <path d="M 35 10 Q 40 22 35 35" stroke="#000000" stroke-width="1.5" fill="none" opacity="0.5"/>
          <!-- Eye (on right side) -->
          <circle cx="58" cy="18" r="2.5" fill="#000000"/>
          <circle cx="57" cy="17" r="1" fill="#ffffff"/>
          <!-- Dorsal fin -->
          <polygon points="45,5 42,2 38,5" fill="#5fd4d8" stroke="#000000" stroke-width="1.5"/>
          <!-- Tail (on left side, fan-shaped) -->
          <path d="M 20 15 L 5 8 L 7 22 L 5 37 L 20 30 Z" fill="#5fd4d8" stroke="#000000" stroke-width="1.5"/>
        </svg>`
       ];
       return fishVariants[index % fishVariants.length];
     }

    /**
     * Load SVG elements from external file (optional, for optimization)
     */
    loadSVGElements() {
      // SVG elements are now inline in the HTML, so this is optional
      // This could load from the external SVG file if needed
    }

    /**
     * Create subtle background for secondary pages
     */
    createSubtleBackground() {
      const bg = document.createElement('div');
      bg.className = 'subtle-ocean-bg';
      document.body.insertBefore(bg, document.body.firstChild);
    }

  /**
   * Create subtle floating bubbles for secondary pages
   */
  createSubtleBubbles() {
    const container = document.createElement('div');
    container.className = 'subtle-bubbles';

    // Create 4 subtle bubbles
    for (let i = 0; i < 4; i++) {
      const bubble = document.createElement('div');
      bubble.className = 'subtle-bubble';
      container.appendChild(bubble);
    }

    document.body.insertBefore(container, document.body.querySelector('.subtle-ocean-bg').nextSibling);
  }

  /**
   * Cleanup function
   */
  destroy() {
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
    }
  }
}

/**
 * Initialize ocean animations when DOM is ready
 */
document.addEventListener('DOMContentLoaded', () => {
  window.oceanManager = new OceanAnimationManager();
});

/**
 * Cleanup on page unload
 */
window.addEventListener('beforeunload', () => {
  if (window.oceanManager) {
    window.oceanManager.destroy();
  }
});

/**
 * Export for use in other modules
 */
if (typeof module !== 'undefined' && module.exports) {
  module.exports = OceanAnimationManager;
}
