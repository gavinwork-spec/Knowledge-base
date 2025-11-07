import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { Theme, ThemeConfig } from '@/types'

interface ThemeStore extends ThemeConfig {
  // Actions
  setTheme: (theme: Theme) => void
  toggleTheme: () => void
  setReducedMotion: (enabled: boolean) => void
  setHighContrast: (enabled: boolean) => void
  setFontSize: (size: 'sm' | 'md' | 'lg' | 'xl') => void
  updateConfig: (config: Partial<ThemeConfig>) => void
  resetTheme: () => void
}

const defaultThemeConfig: ThemeConfig = {
  theme: 'system',
  reducedMotion: false,
  highContrast: false,
  fontSize: 'md',
}

export const useThemeStore = create<ThemeStore>()(
  persist(
    (set, get) => ({
      ...defaultThemeConfig,

      setTheme: (theme) => {
        set({ theme })
        applyThemeToDOM(theme)
      },

      toggleTheme: () => {
        const currentTheme = get().theme
        let newTheme: Theme

        switch (currentTheme) {
          case 'light':
            newTheme = 'dark'
            break
          case 'dark':
            newTheme = 'system'
            break
          case 'system':
          default:
            newTheme = 'light'
            break
        }

        get().setTheme(newTheme)
      },

      setReducedMotion: (enabled) => {
        set({ reducedMotion: enabled })
        applyReducedMotionToDOM(enabled)
      },

      setHighContrast: (enabled) => {
        set({ highContrast: enabled })
        applyHighContrastToDOM(enabled)
      },

      setFontSize: (size) => {
        set({ fontSize: size })
        applyFontSizeToDOM(size)
      },

      updateConfig: (config) => {
        const newConfig = { ...get(), ...config }
        set(newConfig)

        // Apply theme-related changes to DOM
        if (config.theme !== undefined) {
          applyThemeToDOM(config.theme)
        }
        if (config.reducedMotion !== undefined) {
          applyReducedMotionToDOM(config.reducedMotion)
        }
        if (config.highContrast !== undefined) {
          applyHighContrastToDOM(config.highContrast)
        }
        if (config.fontSize !== undefined) {
          applyFontSizeToDOM(config.fontSize)
        }
      },

      resetTheme: () => {
        set(defaultThemeConfig)
        applyThemeToDOM(defaultThemeConfig.theme)
        applyReducedMotionToDOM(defaultThemeConfig.reducedMotion)
        applyHighContrastToDOM(defaultThemeConfig.highContrast)
        applyFontSizeToDOM(defaultThemeConfig.fontSize)
      },
    }),
    {
      name: 'theme-store',
      partialize: (state) => ({
        theme: state.theme,
        reducedMotion: state.reducedMotion,
        highContrast: state.highContrast,
        fontSize: state.fontSize,
      }),
    }
  )
)

// DOM manipulation functions
function applyThemeToDOM(theme: Theme) {
  if (typeof window === 'undefined') return

  const root = window.document.documentElement
  root.classList.remove('light', 'dark')

  if (theme === 'system') {
    const systemTheme = window.matchMedia('(prefers-color-scheme: dark)').matches
      ? 'dark'
      : 'light'
    root.classList.add(systemTheme)
  } else {
    root.classList.add(theme)
  }

  // Update meta theme-color
  const metaThemeColor = document.querySelector('meta[name="theme-color"]')
  if (metaThemeColor) {
    const color = theme === 'dark' ? '#0a0a0a' : '#ffffff'
    metaThemeColor.setAttribute('content', color)
  }
}

function applyReducedMotionToDOM(enabled: boolean) {
  if (typeof window === 'undefined') return

  const root = window.document.documentElement
  if (enabled) {
    root.style.setProperty('--motion-reduce', '1')
  } else {
    root.style.removeProperty('--motion-reduce')
  }
}

function applyHighContrastToDOM(enabled: boolean) {
  if (typeof window === 'undefined') return

  const root = window.document.documentElement
  if (enabled) {
    root.classList.add('high-contrast')
  } else {
    root.classList.remove('high-contrast')
  }
}

function applyFontSizeToDOM(size: 'sm' | 'md' | 'lg' | 'xl') {
  if (typeof window === 'undefined') return

  const root = window.document.documentElement
  const fontSizes = {
    sm: '14px',
    md: '16px',
    lg: '18px',
    xl: '20px',
  }

  root.style.setProperty('--font-size-base', fontSizes[size])
}

// Initialize theme on app startup
export function initializeTheme() {
  if (typeof window === 'undefined') return

  const store = useThemeStore.getState()

  // Apply saved theme to DOM
  applyThemeToDOM(store.theme)
  applyReducedMotionToDOM(store.reducedMotion)
  applyHighContrastToDOM(store.highContrast)
  applyFontSizeToDOM(store.fontSize)

  // Listen for system theme changes
  const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)')
  const handleSystemThemeChange = () => {
    if (store.theme === 'system') {
      applyThemeToDOM('system')
    }
  }

  mediaQuery.addEventListener('change', handleSystemThemeChange)

  // Listen for reduced motion preference changes
  const reducedMotionQuery = window.matchMedia('(prefers-reduced-motion: reduce)')
  const handleReducedMotionChange = (e: MediaQueryListEvent) => {
    // Only auto-update if user hasn't explicitly set a preference
    if (!store.reducedMotion) {
      applyReducedMotionToDOM(e.matches)
    }
  }

  reducedMotionQuery.addEventListener('change', handleReducedMotionChange)

  // Listen for high contrast preference changes
  const highContrastQuery = window.matchMedia('(prefers-contrast: high)')
  const handleHighContrastChange = (e: MediaQueryListEvent) => {
    // Only auto-update if user hasn't explicitly set a preference
    if (!store.highContrast) {
      applyHighContrastToDOM(e.matches)
    }
  }

  highContrastQuery.addEventListener('change', handleHighContrastChange)

  // Cleanup function
  return () => {
    mediaQuery.removeEventListener('change', handleSystemThemeChange)
    reducedMotionQuery.removeEventListener('change', handleReducedMotionChange)
    highContrastQuery.removeEventListener('change', handleHighContrastChange)
  }
}

// Utility hooks
export const useTheme = () => {
  const store = useThemeStore()

  return {
    ...store,
    isDark: store.theme === 'dark' ||
           (store.theme === 'system' &&
            typeof window !== 'undefined' &&
            window.matchMedia('(prefers-color-scheme: dark)').matches),
    isLight: !store.isDark,
  }
}

export const useThemeClass = () => {
  const { theme } = useThemeStore()

  if (typeof window === 'undefined') return 'light'

  if (theme === 'system') {
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
  }

  return theme
}