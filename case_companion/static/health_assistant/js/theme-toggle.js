// Theme Toggle Functionality
(function() {
    'use strict';

    // Get theme from localStorage or default to dark
    const getTheme = () => {
        return localStorage.getItem('curaid-theme') || 'dark';
    };

    // Set theme
    const setTheme = (theme) => {
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem('curaid-theme', theme);
        updateThemeIcon(theme);
    };

    // Update theme toggle icon
    const updateThemeIcon = (theme) => {
        const icon = document.querySelector('.theme-toggle i');
        if (icon) {
            if (theme === 'dark') {
                icon.className = 'fas fa-sun';
            } else {
                icon.className = 'fas fa-moon';
            }
        }
    };

    // Toggle theme
    const toggleTheme = () => {
        const currentTheme = getTheme();
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        setTheme(newTheme);
        
        // Add animation effect
        document.body.classList.add('theme-transition');
        setTimeout(() => {
            document.body.classList.remove('theme-transition');
        }, 300);
    };

    // Initialize theme on page load
    const initTheme = () => {
        const theme = getTheme();
        setTheme(theme);
        
        // Create theme toggle button if it doesn't exist
        if (!document.querySelector('.theme-toggle')) {
            const toggleButton = document.createElement('button');
            toggleButton.className = 'theme-toggle';
            toggleButton.setAttribute('aria-label', 'Toggle theme');
            toggleButton.setAttribute('title', 'Toggle dark/light mode');
            toggleButton.innerHTML = '<i class="fas fa-sun"></i>';
            toggleButton.addEventListener('click', toggleTheme);
            document.body.appendChild(toggleButton);
        }
    };

    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initTheme);
    } else {
        initTheme();
    }

    // Expose toggle function globally if needed
    window.toggleTheme = toggleTheme;
})();
