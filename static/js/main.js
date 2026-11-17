/**
 * ORALVISION AI — ENTERPRISE CLINICAL SCRIPT (v3.0)
 * Global Theme Management, Sidebar Interactions, Telemetry & Micro-Interactions
 */

document.addEventListener('DOMContentLoaded', () => {
    initThemeManager();
    initSidebar();
    initTableSearch();
    initCountUpStats();
    initTooltips();
});

/* --------------------------------------------------------------------------
   1. THEME MANAGER (DARK / LIGHT MODE WITH LOCALSTORAGE PERSISTENCE)
   -------------------------------------------------------------------------- */
function initThemeManager() {
    const savedTheme = localStorage.getItem('oralvision_theme') || 'dark';
    document.documentElement.setAttribute('data-theme', savedTheme);
    updateThemeIcons(savedTheme);

    const toggleBtns = document.querySelectorAll('.theme-toggle-btn');
    toggleBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            document.documentElement.setAttribute('data-theme', newTheme);
            localStorage.setItem('oralvision_theme', newTheme);
            updateThemeIcons(newTheme);
        });
    });
}

function updateThemeIcons(theme) {
    const toggleBtns = document.querySelectorAll('.theme-toggle-btn');
    toggleBtns.forEach(btn => {
        const icon = btn.querySelector('i');
        if (icon) {
            if (theme === 'light') {
                icon.className = 'fas fa-moon';
                btn.title = 'Switch to Clinical Dark Mode';
            } else {
                icon.className = 'fas fa-sun';
                btn.title = 'Switch to Clinical Light Mode';
            }
        }
    });
}

/* --------------------------------------------------------------------------
   2. SIDEBAR NAVIGATION CONTROLS (EXPAND, COLLAPSE, MOBILE SLIDE-OUT)
   -------------------------------------------------------------------------- */
function initSidebar() {
    const sidebar = document.querySelector('.app-sidebar');
    const toggleBtn = document.querySelector('.sidebar-toggle-btn');
    const mobileToggle = document.querySelector('.mobile-nav-toggle');

    if (!sidebar) return;

    // Check saved collapse state
    const isCollapsed = localStorage.getItem('oralvision_sidebar_collapsed') === 'true';
    if (isCollapsed && window.innerWidth > 992) {
        sidebar.classList.add('collapsed');
    }

    if (toggleBtn) {
        toggleBtn.addEventListener('click', () => {
            sidebar.classList.toggle('collapsed');
            const state = sidebar.classList.contains('collapsed');
            localStorage.setItem('oralvision_sidebar_collapsed', state);
            
            const icon = toggleBtn.querySelector('i');
            if (icon) {
                icon.className = state ? 'fas fa-chevron-right' : 'fas fa-chevron-left';
            }
        });
    }

    if (mobileToggle) {
        mobileToggle.addEventListener('click', () => {
            sidebar.classList.toggle('mobile-open');
        });
    }

    // Close mobile sidebar on outside click
    document.addEventListener('click', (e) => {
        if (sidebar.classList.contains('mobile-open') && 
            !sidebar.contains(e.target) && 
            mobileToggle && !mobileToggle.contains(e.target)) {
            sidebar.classList.remove('mobile-open');
        }
    });
}

/* --------------------------------------------------------------------------
   3. CLIENT-SIDE TABLE SEARCH & FILTERING
   -------------------------------------------------------------------------- */
function initTableSearch() {
    const searchInputs = document.querySelectorAll('.table-search-input');
    searchInputs.forEach(input => {
        const targetTableSelector = input.dataset.targetTable || '.table-clinical';
        const table = document.querySelector(targetTableSelector);
        if (!table) return;

        input.addEventListener('input', () => {
            const query = input.value.toLowerCase().trim();
            const rows = table.querySelectorAll('tbody tr');
            
            rows.forEach(row => {
                const text = row.textContent.toLowerCase();
                row.style.display = text.includes(query) ? '' : 'none';
            });
        });
    });
}

/* --------------------------------------------------------------------------
   4. COUNT-UP STATS ANIMATIONS
   -------------------------------------------------------------------------- */
function initCountUpStats() {
    const statElements = document.querySelectorAll('.kpi-value[data-count]');
    statElements.forEach(el => {
        const target = parseInt(el.dataset.count, 10);
        if (isNaN(target)) return;

        let current = 0;
        const duration = 1200;
        const stepTime = Math.abs(Math.floor(duration / (target || 1)));
        
        const timer = setInterval(() => {
            current += 1;
            el.textContent = current;
            if (current >= target) {
                el.textContent = target;
                clearInterval(timer);
            }
        }, Math.max(stepTime, 20));
    });
}

/* --------------------------------------------------------------------------
   5. TOAST NOTIFICATION SYSTEM
   -------------------------------------------------------------------------- */
function showToast(message, type = 'info') {
    let container = document.getElementById('toastContainer');
    if (!container) {
        container = document.createElement('div');
        container.id = 'toastContainer';
        container.style.cssText = 'position:fixed;top:24px;right:24px;z-index:9999;display:flex;flex-direction:column;gap:12px;';
        document.body.appendChild(container);
    }

    const toast = document.createElement('div');
    toast.className = `flash-alert ${type}`;
    toast.style.boxShadow = '0 10px 30px rgba(0,0,0,0.3)';
    toast.style.minWidth = '300px';

    const iconMap = {
        success: 'fa-check-circle',
        error: 'fa-exclamation-triangle',
        warning: 'fa-exclamation-circle',
        info: 'fa-info-circle'
    };

    toast.innerHTML = `
        <i class="fas ${iconMap[type] || 'fa-info-circle'}"></i>
        <div style="flex:1;">${message}</div>
        <button style="background:none;border:none;color:inherit;cursor:pointer;opacity:0.7;" onclick="this.parentElement.remove()">&times;</button>
    `;

    container.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateY(-10px)';
        toast.style.transition = 'all 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

function initTooltips() {
    // Custom tooltips support
}
