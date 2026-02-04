/**
 * ============================================================================
 * FEDERATED LEARNING UI - INTERACTIVE ENHANCEMENTS
 * Scroll Animations | Theme Toggle | Microinteractions | Loading States
 * ============================================================================
 */

(function() {
    'use strict';

    // ========================================================================
    // 1. SCROLL-TRIGGERED ANIMATIONS
    // ========================================================================
    
    class ScrollAnimations {
        constructor() {
            this.observerOptions = {
                threshold: 0.15,
                rootMargin: '0px 0px -50px 0px'
            };
            this.init();
        }

        init() {
            // Create Intersection Observer
            this.observer = new IntersectionObserver(
                this.handleIntersection.bind(this),
                this.observerOptions
            );

            // Observe all elements with animation classes
            this.observeElements();
            
            // Add scroll-based navbar effects
            this.initNavbarScroll();
            
            // Add parallax effects
            this.initParallax();
        }

        observeElements() {
            const animatedElements = document.querySelectorAll(
                '.fade-in, .fade-in-up, .fade-in-down, .fade-in-left, .fade-in-right, ' +
                '.slide-up, .slide-down, .slide-left, .slide-right, ' +
                '.scale-in, .bounce-in, .rotate-in, ' +
                '.stagger-1, .stagger-2, .stagger-3, .stagger-4, .stagger-5, .stagger-6'
            );

            animatedElements.forEach(el => {
                // Add initial state
                el.style.opacity = '0';
                el.style.visibility = 'hidden';
                
                // Observe element
                this.observer.observe(el);
            });
        }

        handleIntersection(entries) {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    // Make element visible and trigger animation
                    entry.target.style.visibility = 'visible';
                    entry.target.style.opacity = '1';
                    entry.target.classList.add('animated');
                    
                    // Unobserve after animation
                    this.observer.unobserve(entry.target);
                }
            });
        }

        initNavbarScroll() {
            const navbar = document.querySelector('.navbar');
            if (!navbar) return;

            let lastScroll = 0;
            
            window.addEventListener('scroll', () => {
                const currentScroll = window.pageYOffset;
                
                // Add scrolled class when scrolled down
                if (currentScroll > 50) {
                    navbar.classList.add('scrolled');
                } else {
                    navbar.classList.remove('scrolled');
                }
                
                // Hide navbar on scroll down, show on scroll up
                if (currentScroll > lastScroll && currentScroll > 100) {
                    navbar.style.transform = 'translateY(-100%)';
                } else {
                    navbar.style.transform = 'translateY(0)';
                }
                
                lastScroll = currentScroll;
            });
        }

        initParallax() {
            const parallaxElements = document.querySelectorAll('.parallax');
            
            if (parallaxElements.length === 0) return;
            
            window.addEventListener('scroll', () => {
                const scrolled = window.pageYOffset;
                
                parallaxElements.forEach(el => {
                    const speed = el.dataset.speed || 0.5;
                    const yPos = -(scrolled * speed);
                    el.style.transform = `translateY(${yPos}px)`;
                });
            });
        }
    }

    // ========================================================================
    // 2. DARK/LIGHT THEME TOGGLE
    // ========================================================================
    
    class ThemeToggle {
        constructor() {
            this.currentTheme = localStorage.getItem('theme') || 'dark';
            this.init();
        }

        init() {
            // Apply saved theme
            this.applyTheme(this.currentTheme);
            
            // Create theme toggle button
            this.createToggleButton();
            
            // Listen for system theme changes
            this.listenToSystemTheme();
        }

        createToggleButton() {
            // Check if button already exists
            if (document.getElementById('theme-toggle')) return;
            
            const navbar = document.querySelector('.navbar-nav');
            if (!navbar) return;

            // Create toggle button
            const toggleItem = document.createElement('li');
            toggleItem.className = 'nav-item';
            
            const toggleButton = document.createElement('button');
            toggleButton.id = 'theme-toggle';
            toggleButton.className = 'btn btn-sm nav-link';
            toggleButton.setAttribute('aria-label', 'Toggle theme');
            toggleButton.style.border = 'none';
            toggleButton.style.background = 'transparent';
            
            // Set initial icon
            this.updateToggleIcon(toggleButton);
            
            // Add click handler
            toggleButton.addEventListener('click', () => this.toggleTheme());
            
            toggleItem.appendChild(toggleButton);
            navbar.appendChild(toggleItem);
        }

        updateToggleIcon(button) {
            const icon = this.currentTheme === 'dark' 
                ? '<i class="fas fa-sun"></i>' 
                : '<i class="fas fa-moon"></i>';
            button.innerHTML = icon;
        }

        toggleTheme() {
            this.currentTheme = this.currentTheme === 'dark' ? 'light' : 'dark';
            this.applyTheme(this.currentTheme);
            
            // Update button icon
            const button = document.getElementById('theme-toggle');
            if (button) this.updateToggleIcon(button);
            
            // Save preference
            localStorage.setItem('theme', this.currentTheme);
            
            // Add transition effect
            document.documentElement.style.transition = 'all 0.3s ease';
            setTimeout(() => {
                document.documentElement.style.transition = '';
            }, 300);
        }

        applyTheme(theme) {
            const root = document.documentElement;
            
            if (theme === 'light') {
                root.style.setProperty('--bg-primary', '#ffffff');
                root.style.setProperty('--bg-secondary', '#f8f9fa');
                root.style.setProperty('--bg-tertiary', '#e9ecef');
                root.style.setProperty('--bg-card', '#ffffff');
                root.style.setProperty('--dark-2', '#f1f3f5');
                root.style.setProperty('--dark-3', '#e9ecef');
                root.style.setProperty('--dark-4', '#dee2e6');
                
                root.style.setProperty('--text-primary', '#1a1a1a');
                root.style.setProperty('--text-secondary', '#495057');
                root.style.setProperty('--text-tertiary', '#6c757d');
                root.style.setProperty('--text-muted', '#adb5bd');
                
                root.style.setProperty('--border-color', 'rgba(0, 0, 0, 0.1)');
                root.style.setProperty('--border-hover', 'rgba(99, 102, 241, 0.5)');
                
                root.style.setProperty('--shadow-sm', '0 2px 4px rgba(0, 0, 0, 0.1)');
                root.style.setProperty('--shadow-md', '0 4px 12px rgba(0, 0, 0, 0.15)');
                root.style.setProperty('--shadow-lg', '0 8px 24px rgba(0, 0, 0, 0.2)');
                root.style.setProperty('--shadow-xl', '0 12px 40px rgba(0, 0, 0, 0.25)');
            } else {
                // Reset to dark theme (default values)
                root.style.setProperty('--bg-primary', '#000000');
                root.style.setProperty('--bg-secondary', '#0a0a0a');
                root.style.setProperty('--bg-tertiary', '#121212');
                root.style.setProperty('--bg-card', '#1a1a1a');
                root.style.setProperty('--dark-2', '#141414');
                root.style.setProperty('--dark-3', '#181818');
                root.style.setProperty('--dark-4', '#0d0d0d');
                
                root.style.setProperty('--text-primary', '#ffffff');
                root.style.setProperty('--text-secondary', '#d1d5db');
                root.style.setProperty('--text-tertiary', '#9ca3af');
                root.style.setProperty('--text-muted', '#6b7280');
                
                root.style.setProperty('--border-color', 'rgba(255, 255, 255, 0.1)');
                root.style.setProperty('--border-hover', 'rgba(99, 102, 241, 0.5)');
                
                root.style.setProperty('--shadow-sm', '0 2px 4px rgba(0, 0, 0, 0.5)');
                root.style.setProperty('--shadow-md', '0 4px 12px rgba(0, 0, 0, 0.6)');
                root.style.setProperty('--shadow-lg', '0 8px 24px rgba(0, 0, 0, 0.7)');
                root.style.setProperty('--shadow-xl', '0 12px 40px rgba(0, 0, 0, 0.8)');
            }
        }

        listenToSystemTheme() {
            if (!window.matchMedia) return;
            
            const darkModeQuery = window.matchMedia('(prefers-color-scheme: dark)');
            
            darkModeQuery.addEventListener('change', (e) => {
                // Only auto-switch if user hasn't manually set a preference
                if (!localStorage.getItem('theme')) {
                    this.currentTheme = e.matches ? 'dark' : 'light';
                    this.applyTheme(this.currentTheme);
                }
            });
        }
    }

    // ========================================================================
    // 3. MICROINTERACTIONS
    // ========================================================================
    
    class Microinteractions {
        constructor() {
            this.init();
        }

        init() {
            this.initButtonRipple();
            this.initCardTilt();
            this.initInputFocus();
            this.initTooltips();
            this.initCountUp();
            this.initSmoothScroll();
        }

        initButtonRipple() {
            const buttons = document.querySelectorAll('.btn');
            
            buttons.forEach(button => {
                button.addEventListener('click', function(e) {
                    // Create ripple element
                    const ripple = document.createElement('span');
                    ripple.className = 'ripple';
                    
                    // Calculate position
                    const rect = this.getBoundingClientRect();
                    const size = Math.max(rect.width, rect.height);
                    const x = e.clientX - rect.left - size / 2;
                    const y = e.clientY - rect.top - size / 2;
                    
                    // Set styles
                    ripple.style.width = ripple.style.height = size + 'px';
                    ripple.style.left = x + 'px';
                    ripple.style.top = y + 'px';
                    
                    // Add to button
                    this.appendChild(ripple);
                    
                    // Remove after animation
                    setTimeout(() => ripple.remove(), 600);
                });
            });
        }

        initCardTilt() {
            const cards = document.querySelectorAll('.card');
            
            cards.forEach(card => {
                card.addEventListener('mousemove', function(e) {
                    const rect = this.getBoundingClientRect();
                    const x = e.clientX - rect.left;
                    const y = e.clientY - rect.top;
                    
                    const centerX = rect.width / 2;
                    const centerY = rect.height / 2;
                    
                    const rotateX = (y - centerY) / 20;
                    const rotateY = (centerX - x) / 20;
                    
                    this.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateY(-6px)`;
                });
                
                card.addEventListener('mouseleave', function() {
                    this.style.transform = '';
                });
            });
        }

        initInputFocus() {
            const inputs = document.querySelectorAll('.form-control');
            
            inputs.forEach(input => {
                // Add floating label effect
                input.addEventListener('focus', function() {
                    const label = this.previousElementSibling;
                    if (label && label.classList.contains('form-label')) {
                        label.style.transform = 'translateY(-2px)';
                        label.style.color = 'var(--primary)';
                    }
                });
                
                input.addEventListener('blur', function() {
                    const label = this.previousElementSibling;
                    if (label && label.classList.contains('form-label')) {
                        label.style.transform = '';
                        label.style.color = '';
                    }
                });
            });
        }

        initTooltips() {
            // Add tooltip functionality to elements with data-tooltip attribute
            const tooltipElements = document.querySelectorAll('[data-tooltip]');
            
            tooltipElements.forEach(el => {
                const tooltipText = el.getAttribute('data-tooltip');
                
                el.addEventListener('mouseenter', function(e) {
                    const tooltip = document.createElement('div');
                    tooltip.className = 'custom-tooltip';
                    tooltip.textContent = tooltipText;
                    tooltip.style.cssText = `
                        position: absolute;
                        background: var(--bg-card);
                        color: var(--text-primary);
                        padding: 0.5rem 1rem;
                        border-radius: 8px;
                        border: 1px solid var(--border-color);
                        font-size: 0.875rem;
                        box-shadow: var(--shadow-lg);
                        z-index: 10000;
                        pointer-events: none;
                        white-space: nowrap;
                        opacity: 0;
                        transition: opacity 0.2s;
                    `;
                    
                    document.body.appendChild(tooltip);
                    
                    // Position tooltip
                    const rect = this.getBoundingClientRect();
                    tooltip.style.left = rect.left + rect.width / 2 - tooltip.offsetWidth / 2 + 'px';
                    tooltip.style.top = rect.top - tooltip.offsetHeight - 8 + 'px';
                    
                    // Fade in
                    setTimeout(() => tooltip.style.opacity = '1', 10);
                    
                    // Store reference for removal
                    this._tooltip = tooltip;
                });
                
                el.addEventListener('mouseleave', function() {
                    if (this._tooltip) {
                        this._tooltip.style.opacity = '0';
                        setTimeout(() => this._tooltip.remove(), 200);
                        this._tooltip = null;
                    }
                });
            });
        }

        initCountUp() {
            const counters = document.querySelectorAll('[data-count]');
            
            counters.forEach(counter => {
                const target = parseInt(counter.getAttribute('data-count'));
                const duration = parseInt(counter.getAttribute('data-duration')) || 2000;
                
                const observer = new IntersectionObserver((entries) => {
                    entries.forEach(entry => {
                        if (entry.isIntersecting) {
                            this.animateCount(counter, 0, target, duration);
                            observer.unobserve(counter);
                        }
                    });
                }, { threshold: 0.5 });
                
                observer.observe(counter);
            });
        }

        animateCount(element, start, end, duration) {
            const range = end - start;
            const increment = range / (duration / 16);
            let current = start;
            
            const timer = setInterval(() => {
                current += increment;
                if (current >= end) {
                    current = end;
                    clearInterval(timer);
                }
                element.textContent = Math.floor(current);
            }, 16);
        }

        initSmoothScroll() {
            const links = document.querySelectorAll('a[href^="#"]');
            
            links.forEach(link => {
                link.addEventListener('click', function(e) {
                    const targetId = this.getAttribute('href');
                    if (targetId === '#') return;
                    
                    const target = document.querySelector(targetId);
                    if (target) {
                        e.preventDefault();
                        target.scrollIntoView({
                            behavior: 'smooth',
                            block: 'start'
                        });
                    }
                });
            });
        }
    }

    // ========================================================================
    // 4. CUSTOM LOADING STATES
    // ========================================================================
    
    class LoadingStates {
        constructor() {
            this.init();
        }

        init() {
            this.createPageLoader();
            this.interceptForms();
            this.interceptLinks();
        }

        createPageLoader() {
            // Create full-page loader
            const loader = document.createElement('div');
            loader.id = 'page-loader';
            loader.innerHTML = `
                <div class="loader-content">
                    <div class="spinner-lg"></div>
                    <p style="margin-top: 1rem; color: var(--text-secondary);">Loading...</p>
                </div>
            `;
            loader.style.cssText = `
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: var(--bg-primary);
                display: flex;
                align-items: center;
                justify-content: center;
                z-index: 99999;
                opacity: 1;
                transition: opacity 0.3s ease;
            `;
            
            document.body.appendChild(loader);
            
            // Hide loader when page is loaded
            window.addEventListener('load', () => {
                setTimeout(() => {
                    loader.style.opacity = '0';
                    setTimeout(() => loader.remove(), 300);
                }, 300);
            });
        }

        showButtonLoader(button) {
            if (button._loading) return;
            
            button._loading = true;
            button._originalContent = button.innerHTML;
            button.disabled = true;
            
            button.innerHTML = `
                <span class="spinner" style="width: 20px; height: 20px; border-width: 2px; margin-right: 0.5rem;"></span>
                <span>Loading...</span>
            `;
        }

        hideButtonLoader(button) {
            if (!button._loading) return;
            
            button._loading = false;
            button.disabled = false;
            button.innerHTML = button._originalContent;
        }

        interceptForms() {
            const forms = document.querySelectorAll('form');
            
            forms.forEach(form => {
                form.addEventListener('submit', (e) => {
                    const submitButton = form.querySelector('button[type="submit"]');
                    if (submitButton) {
                        this.showButtonLoader(submitButton);
                    }
                });
            });
        }

        interceptLinks() {
            // Add loading state to navigation links
            const navLinks = document.querySelectorAll('.nav-link, .btn[href]');
            
            navLinks.forEach(link => {
                link.addEventListener('click', function(e) {
                    // Skip for hash links and external links
                    const href = this.getAttribute('href');
                    if (!href || href.startsWith('#') || href.startsWith('http')) return;
                    
                    // Add loading indicator
                    const loader = document.createElement('div');
                    loader.className = 'page-transition-loader';
                    loader.style.cssText = `
                        position: fixed;
                        top: 0;
                        left: 0;
                        width: 0%;
                        height: 3px;
                        background: linear-gradient(90deg, var(--primary), var(--accent));
                        z-index: 99999;
                        transition: width 0.5s ease;
                    `;
                    document.body.appendChild(loader);
                    
                    setTimeout(() => loader.style.width = '80%', 10);
                });
            });
        }

        showSkeletonLoader(container) {
            container.innerHTML = `
                <div class="skeleton-loader">
                    <div class="skeleton-line" style="width: 60%; height: 20px; margin-bottom: 1rem;"></div>
                    <div class="skeleton-line" style="width: 100%; height: 15px; margin-bottom: 0.5rem;"></div>
                    <div class="skeleton-line" style="width: 100%; height: 15px; margin-bottom: 0.5rem;"></div>
                    <div class="skeleton-line" style="width: 80%; height: 15px;"></div>
                </div>
            `;
        }
    }

    // ========================================================================
    // 5. PAGE TRANSITION EFFECTS
    // ========================================================================
    
    class PageTransitions {
        constructor() {
            this.init();
        }

        init() {
            this.addPageEnterAnimation();
            this.addPageExitAnimation();
        }

        addPageEnterAnimation() {
            // Add fade-in effect to page content
            document.body.style.opacity = '0';
            
            window.addEventListener('load', () => {
                setTimeout(() => {
                    document.body.style.transition = 'opacity 0.3s ease';
                    document.body.style.opacity = '1';
                }, 100);
            });
        }

        addPageExitAnimation() {
            // Intercept all internal links for smooth transitions
            document.addEventListener('click', (e) => {
                const link = e.target.closest('a');
                
                if (!link) return;
                
                const href = link.getAttribute('href');
                
                // Skip external links, hash links, and links with target attribute
                if (!href || 
                    href.startsWith('#') || 
                    href.startsWith('http') || 
                    link.hasAttribute('target') ||
                    e.ctrlKey || 
                    e.metaKey) {
                    return;
                }
                
                // Prevent default and add transition
                e.preventDefault();
                
                // Fade out content
                document.body.style.transition = 'opacity 0.2s ease';
                document.body.style.opacity = '0';
                
                // Navigate after animation
                setTimeout(() => {
                    window.location.href = href;
                }, 200);
            });
        }
    }

    // ========================================================================
    // 6. MOBILE MENU ENHANCEMENTS
    // ========================================================================
    
    class MobileMenu {
        constructor() {
            this.init();
        }

        init() {
            this.createBackdrop();
            this.setupMenuToggle();
            this.setupMenuClose();
        }

        createBackdrop() {
            // Create backdrop element
            const backdrop = document.createElement('div');
            backdrop.className = 'mobile-backdrop';
            backdrop.id = 'mobile-backdrop';
            document.body.appendChild(backdrop);

            // Close menu when backdrop is clicked
            backdrop.addEventListener('click', () => {
                this.closeMenu();
            });
        }

        setupMenuToggle() {
            const toggler = document.querySelector('.navbar-toggler');
            const navbarCollapse = document.querySelector('.navbar-collapse');

            if (!toggler || !navbarCollapse) return;

            toggler.addEventListener('click', () => {
                const isOpen = navbarCollapse.classList.contains('show');
                
                if (isOpen) {
                    this.closeMenu();
                } else {
                    this.openMenu();
                }
            });
        }

        setupMenuClose() {
            const navbarCollapse = document.querySelector('.navbar-collapse');
            if (!navbarCollapse) return;

            // Close on navigation link click
            const navLinks = navbarCollapse.querySelectorAll('.nav-link');
            navLinks.forEach(link => {
                link.addEventListener('click', () => {
                    setTimeout(() => this.closeMenu(), 200);
                });
            });

            // Close button (using CSS ::before pseudo-element)
            navbarCollapse.addEventListener('click', (e) => {
                const rect = navbarCollapse.getBoundingClientRect();
                const closeButtonArea = {
                    left: rect.right - 70,
                    top: rect.top,
                    right: rect.right - 10,
                    bottom: rect.top + 70
                };

                if (e.clientX >= closeButtonArea.left &&
                    e.clientX <= closeButtonArea.right &&
                    e.clientY >= closeButtonArea.top &&
                    e.clientY <= closeButtonArea.bottom) {
                    this.closeMenu();
                }
            });

            // Close on Escape key
            document.addEventListener('keydown', (e) => {
                if (e.key === 'Escape' && navbarCollapse.classList.contains('show')) {
                    this.closeMenu();
                }
            });
        }

        openMenu() {
            const navbarCollapse = document.querySelector('.navbar-collapse');
            const backdrop = document.getElementById('mobile-backdrop');
            
            if (navbarCollapse && backdrop) {
                navbarCollapse.classList.add('show');
                backdrop.style.display = 'block';
                document.body.style.overflow = 'hidden'; // Prevent background scrolling
            }
        }

        closeMenu() {
            const navbarCollapse = document.querySelector('.navbar-collapse');
            const backdrop = document.getElementById('mobile-backdrop');
            
            if (navbarCollapse && backdrop) {
                navbarCollapse.classList.remove('show');
                backdrop.style.display = 'none';
                document.body.style.overflow = ''; // Restore scrolling
            }
        }
    }

    // ========================================================================
    // INITIALIZE ALL FEATURES
    // ========================================================================
    
    document.addEventListener('DOMContentLoaded', () => {
        // Initialize all features
        new ScrollAnimations();
        new ThemeToggle();
        new Microinteractions();
        new LoadingStates();
        new PageTransitions();
        new MobileMenu();
        
        console.log('✅ Federated Learning UI Enhancements Initialized');
    });

    // ========================================================================
    // GLOBAL UTILITY FUNCTIONS
    // ========================================================================
    
    window.FL = {
        // Show toast notification
        showToast: function(message, type = 'info', duration = 3000) {
            const toast = document.createElement('div');
            toast.className = `toast-notification toast-${type}`;
            toast.innerHTML = `
                <i class="fas ${this.getToastIcon(type)}"></i>
                <span>${message}</span>
            `;
            toast.style.cssText = `
                position: fixed;
                bottom: 2rem;
                right: 2rem;
                padding: 1rem 1.5rem;
                background: var(--bg-card);
                border: 1px solid var(--border-color);
                border-radius: 12px;
                box-shadow: var(--shadow-xl);
                display: flex;
                align-items: center;
                gap: 0.75rem;
                z-index: 99999;
                animation: slideInRight 0.3s ease;
                max-width: 400px;
            `;
            
            // Add color based on type
            const colors = {
                success: 'var(--success)',
                error: 'var(--danger)',
                warning: 'var(--warning)',
                info: 'var(--info)'
            };
            toast.style.borderLeftWidth = '4px';
            toast.style.borderLeftColor = colors[type] || colors.info;
            
            document.body.appendChild(toast);
            
            // Auto remove
            setTimeout(() => {
                toast.style.animation = 'slideOutRight 0.3s ease';
                setTimeout(() => toast.remove(), 300);
            }, duration);
        },
        
        getToastIcon: function(type) {
            const icons = {
                success: 'fa-check-circle',
                error: 'fa-exclamation-circle',
                warning: 'fa-exclamation-triangle',
                info: 'fa-info-circle'
            };
            return icons[type] || icons.info;
        },
        
        // Show/hide loading spinner on element
        showLoader: function(element) {
            element.classList.add('loading');
            element.style.position = 'relative';
            element.style.pointerEvents = 'none';
            element.style.opacity = '0.6';
            
            const loader = document.createElement('div');
            loader.className = 'element-loader';
            loader.innerHTML = '<div class="spinner"></div>';
            loader.style.cssText = `
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                z-index: 10;
            `;
            element.appendChild(loader);
        },
        
        hideLoader: function(element) {
            element.classList.remove('loading');
            element.style.pointerEvents = '';
            element.style.opacity = '';
            
            const loader = element.querySelector('.element-loader');
            if (loader) loader.remove();
        }
    };

})();
