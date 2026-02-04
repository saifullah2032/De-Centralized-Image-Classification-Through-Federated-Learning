# JavaScript Enhancements Documentation

## Overview

This document describes the interactive enhancements added to the Federated Learning web application. These features provide a modern, professional user experience with smooth animations, theme switching, and responsive feedback.

---

## Table of Contents

1. [Scroll-Triggered Animations](#1-scroll-triggered-animations)
2. [Dark/Light Theme Toggle](#2-darklight-theme-toggle)
3. [Microinteractions](#3-microinteractions)
4. [Custom Loading States](#4-custom-loading-states)
5. [Page Transitions](#5-page-transitions)
6. [Global Utility Functions](#6-global-utility-functions)

---

## 1. Scroll-Triggered Animations

### Description
Automatically animates elements as they come into view while scrolling.

### Usage

Add animation classes to any HTML element:

```html
<div class="fade-in">This will fade in on scroll</div>
<div class="slide-up">This will slide up on scroll</div>
<div class="stagger-1">First item</div>
<div class="stagger-2">Second item (delayed)</div>
<div class="stagger-3">Third item (more delay)</div>
```

### Available Animation Classes

**Fade Animations:**
- `.fade-in` - Simple fade in
- `.fade-in-up` - Fade in while sliding up
- `.fade-in-down` - Fade in while sliding down
- `.fade-in-left` - Fade in from left
- `.fade-in-right` - Fade in from right

**Slide Animations:**
- `.slide-up` - Slide from bottom
- `.slide-down` - Slide from top
- `.slide-left` - Slide from right
- `.slide-right` - Slide from left

**Scale Animations:**
- `.scale-in` - Scale from 95% to 100%
- `.scale-in-center` - Scale from 80% to 100%

**Other Animations:**
- `.bounce-in` - Bounce entrance
- `.rotate-in` - Rotate entrance
- `.stagger-1` through `.stagger-6` - Sequential reveals

### Parallax Effect

Add parallax scrolling effect:

```html
<div class="parallax" data-speed="0.5">
    <img src="background.jpg" alt="Background">
</div>
```

`data-speed`: Controls parallax speed (0.1 to 1.0). Default: 0.5

### Navbar Scroll Effects

The navbar automatically:
- Adds `.scrolled` class when scrolled 50px down
- Hides when scrolling down (after 100px)
- Shows when scrolling up
- Smooth transitions for all states

---

## 2. Dark/Light Theme Toggle

### Description
Toggle between dark and light themes with persistent storage.

### Features

- **Automatic Toggle Button**: Appears in the navbar
- **Local Storage**: Saves user preference
- **System Theme Detection**: Respects OS dark mode preference
- **Smooth Transitions**: Animated color changes

### Usage

The theme toggle is automatically created in the navbar. Users can click the sun/moon icon to switch themes.

### Programmatic Control

```javascript
// Access theme system
const themeToggle = new ThemeToggle();

// Apply specific theme
themeToggle.applyTheme('light');
themeToggle.applyTheme('dark');

// Toggle theme
themeToggle.toggleTheme();
```

### Theme Variables

The following CSS variables change with theme:

**Dark Theme (Default):**
- Background: Black tones (#000000, #0a0a0a, #121212)
- Text: White/gray (#ffffff, #d1d5db, #9ca3af)
- Borders: Subtle white (rgba(255, 255, 255, 0.1))

**Light Theme:**
- Background: White tones (#ffffff, #f8f9fa, #e9ecef)
- Text: Dark gray (#1a1a1a, #495057, #6c757d)
- Borders: Subtle black (rgba(0, 0, 0, 0.1))

---

## 3. Microinteractions

### Description
Small, delightful interactions that enhance user experience.

### Features

#### 3.1 Button Ripple Effect

Automatically applied to all `.btn` elements. Creates Material Design-style ripple on click.

```html
<button class="btn btn-primary">Click me</button>
```

#### 3.2 Card Tilt Effect

3D tilt effect on mouse movement over cards.

```html
<div class="card">
    <div class="card-body">
        Hover over me!
    </div>
</div>
```

#### 3.3 Input Focus Animations

Labels animate when input is focused.

```html
<label class="form-label">Username</label>
<input type="text" class="form-control">
```

#### 3.4 Custom Tooltips

Add tooltips to any element:

```html
<button data-tooltip="Click to submit">Submit</button>
<span data-tooltip="More information">Hover me</span>
```

#### 3.5 Count-Up Animation

Animate numbers counting up:

```html
<div data-count="1000" data-duration="2000">0</div>
```

- `data-count`: Target number
- `data-duration`: Animation duration in milliseconds (default: 2000)

#### 3.6 Smooth Scroll

Automatically enabled for all anchor links:

```html
<a href="#section-2">Jump to Section 2</a>
<div id="section-2">Content here</div>
```

---

## 4. Custom Loading States

### Description
Professional loading indicators for async operations.

### Features

#### 4.1 Page Loader

Automatically shown on page load and hidden when content is ready.

#### 4.2 Button Loading State

Automatically applied to form submit buttons:

```html
<form>
    <button type="submit" class="btn btn-primary">Submit</button>
</form>
```

When submitted, button shows spinner and "Loading..." text.

#### 4.3 Element Loading Overlay

Show loading state on any element:

```javascript
// Show loader
FL.showLoader(document.querySelector('#my-element'));

// Hide loader
FL.hideLoader(document.querySelector('#my-element'));
```

#### 4.4 Skeleton Loader

Display skeleton loader while content loads:

```javascript
const container = document.querySelector('#content');
const loader = new LoadingStates();
loader.showSkeletonLoader(container);

// After content loads, replace with actual content
container.innerHTML = actualContent;
```

#### 4.5 Progress Bar Loader

Shown during page navigation. Automatically appears when clicking internal links.

---

## 5. Page Transitions

### Description
Smooth fade transitions between pages.

### Features

- **Page Enter**: Fade in when page loads
- **Page Exit**: Fade out before navigation
- **Automatic**: Applied to all internal links

### Behavior

1. User clicks internal link
2. Page fades out (200ms)
3. Navigation occurs
4. New page fades in (300ms)

### Disable for Specific Links

Add `target="_blank"` or use external URLs to disable transitions:

```html
<a href="https://external.com">External link (no transition)</a>
<a href="/internal" target="_blank">Internal in new tab (no transition)</a>
```

---

## 6. Global Utility Functions

### Description
Helper functions available globally via `window.FL` object.

### 6.1 Show Toast Notification

Display temporary notifications:

```javascript
// Success notification
FL.showToast('Operation completed successfully!', 'success', 3000);

// Error notification
FL.showToast('An error occurred', 'error', 5000);

// Warning notification
FL.showToast('Please check your input', 'warning');

// Info notification (default)
FL.showToast('New updates available', 'info');
```

**Parameters:**
- `message` (string): Notification text
- `type` (string): 'success', 'error', 'warning', or 'info' (default: 'info')
- `duration` (number): Display time in milliseconds (default: 3000)

### 6.2 Show/Hide Element Loader

Display loading spinner on specific elements:

```javascript
const myElement = document.querySelector('#my-card');

// Show loader
FL.showLoader(myElement);

// Perform async operation
await fetchData();

// Hide loader
FL.hideLoader(myElement);
```

### Example Usage

```javascript
// Complete example with async operation
async function loadUserData() {
    const card = document.querySelector('#user-card');
    
    try {
        FL.showLoader(card);
        const response = await fetch('/api/user');
        const data = await response.json();
        card.innerHTML = renderUserData(data);
        FL.showToast('User data loaded', 'success');
    } catch (error) {
        FL.showToast('Failed to load user data', 'error');
    } finally {
        FL.hideLoader(card);
    }
}
```

---

## Browser Compatibility

All features are compatible with:
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

### Polyfills

The following modern APIs are used:
- `IntersectionObserver` - For scroll animations
- `matchMedia` - For system theme detection
- `localStorage` - For theme persistence
- `scrollIntoView` - For smooth scrolling

---

## Accessibility

### Reduced Motion Support

Users who prefer reduced motion (via OS settings) will see instant animations:

```css
@media (prefers-reduced-motion: reduce) {
    * {
        animation-duration: 0.01ms !important;
        transition-duration: 0.01ms !important;
    }
}
```

### Focus Indicators

All interactive elements have visible focus indicators:
- 2px outline in primary color
- 2px offset for clarity

### Keyboard Navigation

All features support keyboard navigation:
- Theme toggle: Tab to focus, Enter/Space to activate
- Smooth scroll: Works with Tab navigation
- Forms: Standard keyboard controls

---

## Performance Considerations

### Debouncing

Scroll events are optimized with passive listeners and RAF (RequestAnimationFrame).

### Lazy Observation

Intersection Observer only monitors elements with animation classes, reducing overhead.

### CSS-Based Animations

All animations use CSS transitions/animations, hardware-accelerated when possible.

### Local Storage

Theme preference is cached to avoid FOUC (Flash of Unstyled Content).

---

## Customization

### Animation Timing

Modify animation durations in CSS:

```css
/* In custom.css or animations.css */
.fade-in {
    animation-duration: 0.8s; /* Change from default 0.6s */
}
```

### Theme Colors

Modify theme variables in JavaScript:

```javascript
// In app.js, modify ThemeToggle.applyTheme() method
root.style.setProperty('--primary', '#your-color');
```

### Disable Specific Features

Comment out feature initialization in app.js:

```javascript
document.addEventListener('DOMContentLoaded', () => {
    new ScrollAnimations();      // Keep
    // new ThemeToggle();        // Disable theme toggle
    new Microinteractions();     // Keep
    // new LoadingStates();      // Disable loading states
    new PageTransitions();       // Keep
});
```

---

## Troubleshooting

### Animations Not Working

1. Check that `animations.css` is loaded
2. Verify elements have correct animation classes
3. Check browser console for JavaScript errors
4. Ensure `app.js` is loaded after DOM content

### Theme Toggle Not Appearing

1. Verify `.navbar-nav` element exists
2. Check that Font Awesome icons are loaded
3. Ensure JavaScript is not blocked

### Loading States Not Showing

1. Check that buttons have `type="submit"` attribute
2. Verify forms are properly structured
3. Ensure `app.js` is loaded before form submission

### Page Transitions Causing Issues

Disable for specific links:
```html
<a href="/page" onclick="event.stopPropagation()">No transition</a>
```

---

## Examples

### Complete Card with All Features

```html
<div class="card fade-in-up" data-tooltip="Click for more info">
    <div class="card-body">
        <h3 class="card-title">
            <i class="fas fa-chart-line"></i>
            Statistics
        </h3>
        <div class="stat-value" data-count="1250" data-duration="2000">0</div>
        <p class="card-text">Total Users</p>
        <button class="btn btn-primary">
            <i class="fas fa-arrow-right"></i>
            View Details
        </button>
    </div>
</div>
```

### Async Form with Loading State

```html
<form id="user-form">
    <div class="form-group">
        <label class="form-label">Username</label>
        <input type="text" class="form-control" required>
    </div>
    <button type="submit" class="btn btn-primary">
        <i class="fas fa-save"></i> Save
    </button>
</form>

<script>
document.getElementById('user-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const button = e.target.querySelector('button[type="submit"]');
    
    try {
        // Button loading state is automatic
        await fetch('/api/user', {
            method: 'POST',
            body: new FormData(e.target)
        });
        FL.showToast('User saved successfully!', 'success');
    } catch (error) {
        FL.showToast('Failed to save user', 'error');
    }
});
</script>
```

### Landing Page with Staggered Animations

```html
<div class="hero-section fade-in">
    <h1 class="slide-up">Welcome to Federated Learning</h1>
    <p class="lead fade-in-up delay-200">Train AI models without sharing data</p>
</div>

<div class="row">
    <div class="col-md-4 stagger-1">
        <div class="card">Feature 1</div>
    </div>
    <div class="col-md-4 stagger-2">
        <div class="card">Feature 2</div>
    </div>
    <div class="col-md-4 stagger-3">
        <div class="card">Feature 3</div>
    </div>
</div>
```

---

## Support

For issues or questions:
1. Check this documentation
2. Review browser console for errors
3. Verify all dependencies are loaded
4. Check HTML structure matches examples

---

## Credits

**Developer:** Federated Learning Team  
**Version:** 1.0.0  
**Last Updated:** 2024

**Technologies Used:**
- Vanilla JavaScript (ES6+)
- CSS3 Animations
- Intersection Observer API
- Local Storage API
- Bootstrap 5.3

---

## License

This code is part of the Federated Learning project.
