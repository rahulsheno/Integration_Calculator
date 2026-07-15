import { useState } from 'react';
import { NavLink, Outlet, useLocation } from 'react-router-dom';
import { useApp } from '../context/AppContext';
import {
  Calculator, Library, FolderOpen, Moon, Sun, Menu, X,
  Keyboard
} from 'lucide-react';

const NAV_ITEMS = [
  { to: '/solve', label: 'Solver', icon: Calculator },
  { to: '/library', label: 'Library', icon: Library },
  { to: '/collections', label: 'Collections', icon: FolderOpen },
];

export default function Layout() {
  const { theme, toggleTheme } = useApp();
  const location = useLocation();
  const [mobileOpen, setMobileOpen] = useState(false);

  const sidebar = (
    <div className="flex flex-col h-full">
      <div className="p-4 border-b border-[var(--border)]">
        <NavLink to="/" className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-primary-500 flex items-center justify-center">
            <span className="text-white font-bold text-sm">∫</span>
          </div>
          <div>
            <div className="font-semibold text-sm">Calculus Solver</div>
            <div className="text-[10px] text-[var(--muted)]">& Checker</div>
          </div>
        </NavLink>
      </div>

      <nav className="flex-1 p-3 space-y-1">
        {NAV_ITEMS.map(item => (
          <NavLink
            key={item.to}
            to={item.to}
            onClick={() => setMobileOpen(false)}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                isActive
                  ? 'bg-primary-500/10 text-primary-600 dark:text-primary-400'
                  : 'text-[var(--muted)] hover:text-[var(--fg)] hover:bg-[var(--bg)]'
              }`
            }
          >
            <item.icon className="w-4 h-4" />
            {item.label}
          </NavLink>
        ))}
      </nav>

      <div className="p-3 border-t border-[var(--border)] space-y-2">
        <button
          onClick={toggleTheme}
          className="flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm text-[var(--muted)] hover:text-[var(--fg)] hover:bg-[var(--bg)] transition-colors w-full"
        >
          {theme === 'dark' ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
          {theme === 'dark' ? 'Light Mode' : 'Dark Mode'}
        </button>
        <div className="px-3 py-1 text-[10px] text-[var(--muted)] text-center">
          Press <kbd className="px-1 py-0.5 rounded bg-[var(--bg)] border border-[var(--border)] text-[10px]">Ctrl+K</kbd> to search
        </div>
      </div>
    </div>
  );

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Desktop sidebar */}
      <aside className="hidden md:flex w-56 flex-shrink-0 border-r border-[var(--border)] bg-[var(--card)]">
        {sidebar}
      </aside>

      {/* Mobile sidebar */}
      {mobileOpen && (
        <>
          <div className="fixed inset-0 bg-black/50 z-40 md:hidden" onClick={() => setMobileOpen(false)} />
          <aside className="fixed inset-y-0 left-0 w-56 z-50 bg-[var(--card)] border-r border-[var(--border)] md:hidden">
            {sidebar}
          </aside>
        </>
      )}

      {/* Main */}
      <div className="flex-1 flex flex-col min-w-0 overflow-auto">
        {/* Mobile header */}
        <div className="md:hidden flex items-center gap-3 p-4 border-b border-[var(--border)] bg-[var(--card)] sticky top-0 z-30">
          <button onClick={() => setMobileOpen(true)} className="p-1">
            <Menu className="w-5 h-5" />
          </button>
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-lg bg-primary-500 flex items-center justify-center">
              <span className="text-white font-bold text-xs">∫</span>
            </div>
            <span className="font-semibold text-sm">Calculus Solver</span>
          </div>
        </div>

        {/* Page content */}
        <main className="flex-1 p-4 md:p-6 overflow-y-auto">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
