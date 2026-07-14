/**
 * main.js — Core Application JavaScript
 * =======================================
 * Handles:
 *   - Dark mode toggle with localStorage persistence
 *   - Smooth scroll navigation
 *   - Mobile menu toggle
 *   - Navbar scroll behavior
 *   - Page animations
 */

document.addEventListener('DOMContentLoaded', () => {
    initThemeToggle();
    initNavbarScroll();
    initMobileMenu();
    initSmoothScroll();
    initAnimations();
});


// ---------------------------------------------------------------------------
// Dark Mode Toggle
// ---------------------------------------------------------------------------

function initThemeToggle() {
    const toggle = document.getElementById('themeToggle');
    const icon = document.getElementById('themeIcon');
    const html = document.documentElement;

    // Load saved preference or default to 'light'
    const savedTheme = localStorage.getItem('pancreasai-theme') || 'light';
    html.setAttribute('data-theme', savedTheme);
    updateThemeIcon(savedTheme, icon);

    if (toggle) {
        toggle.addEventListener('click', () => {
            const current = html.getAttribute('data-theme');
            const next = current === 'dark' ? 'light' : 'dark';

            html.setAttribute('data-theme', next);
            localStorage.setItem('pancreasai-theme', next);
            updateThemeIcon(next, icon);
        });
    }
}

function updateThemeIcon(theme, iconEl) {
    if (!iconEl) return;
    if (theme === 'dark') {
        iconEl.className = 'fas fa-sun';
    } else {
        iconEl.className = 'fas fa-moon';
    }
}


// ---------------------------------------------------------------------------
// Navbar Scroll Behavior
// ---------------------------------------------------------------------------

function initNavbarScroll() {
    const navbar = document.getElementById('navbar');
    if (!navbar) return;

    let lastScroll = 0;

    window.addEventListener('scroll', () => {
        const currentScroll = window.pageYOffset;

        // Add shadow when scrolled
        if (currentScroll > 20) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }

        lastScroll = currentScroll;
    }, { passive: true });
}


// ---------------------------------------------------------------------------
// Mobile Menu
// ---------------------------------------------------------------------------

function initMobileMenu() {
    const btn = document.getElementById('mobileMenuBtn');
    const links = document.getElementById('navLinks');

    if (btn && links) {
        btn.addEventListener('click', () => {
            links.classList.toggle('active');

            // Update icon
            const icon = btn.querySelector('i');
            if (links.classList.contains('active')) {
                icon.className = 'fas fa-times';
            } else {
                icon.className = 'fas fa-bars';
            }
        });

        // Close menu on link click
        links.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', () => {
                links.classList.remove('active');
                const icon = btn.querySelector('i');
                icon.className = 'fas fa-bars';
            });
        });
    }
}


// ---------------------------------------------------------------------------
// Smooth Scroll
// ---------------------------------------------------------------------------

function initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            const targetId = this.getAttribute('href');
            if (targetId === '#') return;

            const target = document.querySelector(targetId);
            if (target) {
                e.preventDefault();
                const navHeight = document.getElementById('navbar')?.offsetHeight || 72;
                const targetPosition = target.getBoundingClientRect().top + window.pageYOffset - navHeight;

                window.scrollTo({
                    top: targetPosition,
                    behavior: 'smooth'
                });
            }
        });
    });
}


// ---------------------------------------------------------------------------
// Scroll Animations (Intersection Observer)
// ---------------------------------------------------------------------------

function initAnimations() {
    // Animate elements when they come into view
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate-in');
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);

    // Observe feature cards, pipeline steps, and metric cards
    const animateElements = document.querySelectorAll(
        '.feature-card, .pipeline-step, .metric-card, .dashboard-block, .image-card'
    );

    animateElements.forEach((el, index) => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(20px)';
        el.style.transition = `all 0.5s ease ${index * 0.05}s`;
        observer.observe(el);
    });
}

// CSS class for animated elements
const style = document.createElement('style');
style.textContent = `
    .animate-in {
        opacity: 1 !important;
        transform: translateY(0) !important;
    }
`;
document.head.appendChild(style);


// ---------------------------------------------------------------------------
// Loading Overlay Controller
// ---------------------------------------------------------------------------

/**
 * Show the loading overlay and animate through pipeline steps.
 */
function showLoading() {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) {
        overlay.classList.add('active');
        animateLoadingSteps();
    }
}

/**
 * Hide the loading overlay.
 */
function hideLoading() {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) {
        overlay.classList.remove('active');
    }
}

/**
 * Animate through the loading pipeline steps sequentially.
 */
function animateLoadingSteps() {
    const steps = document.querySelectorAll('.loading-step');
    let currentStep = 0;

    const interval = setInterval(() => {
        if (currentStep > 0 && currentStep <= steps.length) {
            // Mark previous step as done
            const prev = steps[currentStep - 1];
            prev.classList.remove('active');
            prev.classList.add('done');
            const prevIcon = prev.querySelector('i');
            prevIcon.className = 'fas fa-check-circle';
        }

        if (currentStep < steps.length) {
            // Activate current step
            steps[currentStep].classList.add('active');
            const icon = steps[currentStep].querySelector('i');
            icon.className = 'fas fa-circle-notch fa-spin';
        } else {
            clearInterval(interval);
        }

        currentStep++;
    }, 1500);
}


// ---------------------------------------------------------------------------
// Notification System
// ---------------------------------------------------------------------------

/**
 * Show a toast notification.
 * @param {string} message - Notification text.
 * @param {string} type - 'success', 'error', or 'info'.
 */
function showNotification(message, type = 'info') {
    const container = document.querySelector('.flash-container') || createNotificationContainer();

    const icons = {
        success: 'fa-check-circle',
        error: 'fa-exclamation-circle',
        info: 'fa-info-circle'
    };

    const notification = document.createElement('div');
    notification.className = `flash-message flash-${type}`;
    notification.innerHTML = `
        <i class="fas ${icons[type] || icons.info}"></i>
        <span>${message}</span>
        <button class="flash-close" onclick="this.parentElement.remove()">
            <i class="fas fa-times"></i>
        </button>
    `;

    container.appendChild(notification);

    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (notification.parentElement) {
            notification.style.animation = 'slideIn 0.3s ease reverse';
            setTimeout(() => notification.remove(), 300);
        }
    }, 5000);
}

function createNotificationContainer() {
    const container = document.createElement('div');
    container.className = 'flash-container';
    document.body.appendChild(container);
    return container;
}
