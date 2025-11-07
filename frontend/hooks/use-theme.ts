import { useEffect } from 'react'
import { useTheme, useThemeClass } from '@/store/theme-store'

/**
 * Hook to handle theme-related functionality
 */
export const useThemeHook = () => {
  const theme = useTheme()
  const themeClass = useThemeClass()

  useEffect(() => {
    // Apply theme-related classes to body
    document.body.classList.toggle('dark', theme.isDark)
    document.body.classList.toggle('light', theme.isLight)
    document.body.classList.toggle('reduced-motion', theme.reducedMotion)
    document.body.classList.toggle('high-contrast', theme.highContrast)

    // Apply font size class
    document.body.classList.remove('text-sm', 'text-md', 'text-lg', 'text-xl')
    document.body.classList.add(`text-${theme.fontSize}`)
  }, [theme])

  return {
    ...theme,
    themeClass,
    currentTheme: themeClass,
    isDark: theme.isDark,
    isLight: theme.isLight,
  }
}