import * as React from 'react'
import { cn } from '@/lib/utils'

interface LoadingSpinnerProps extends React.HTMLAttributes<HTMLDivElement> {
  size?: 'sm' | 'md' | 'lg' | 'xl'
  variant?: 'default' | 'primary' | 'secondary'
}

const LoadingSpinner = React.forwardRef<HTMLDivElement, LoadingSpinnerProps>(
  ({ className, size = 'md', variant = 'default', ...props }, ref) => {
    const sizeClasses = {
      sm: 'w-4 h-4',
      md: 'w-6 h-6',
      lg: 'w-8 h-8',
      xl: 'w-12 h-12',
    }

    const variantClasses = {
      default: 'border-muted-foreground border-t-current',
      primary: 'border-primary/20 border-t-primary',
      secondary: 'border-secondary/20 border-t-secondary',
    }

    return (
      <div
        ref={ref}
        className={cn(
          'animate-spin rounded-full border-2',
          sizeClasses[size],
          variantClasses[variant],
          className
        )}
        {...props}
      />
    )
  }
)
LoadingSpinner.displayName = 'LoadingSpinner'

interface LoadingDotsProps extends React.HTMLAttributes<HTMLDivElement> {
  size?: 'sm' | 'md' | 'lg'
}

const LoadingDots = React.forwardRef<HTMLDivElement, LoadingDotsProps>(
  ({ className, size = 'md', ...props }, ref) => {
    const sizeClasses = {
      sm: 'text-sm',
      md: 'text-base',
      lg: 'text-lg',
    }

    return (
      <div
        ref={ref}
        className={cn('flex items-center space-x-1', sizeClasses[size], className)}
        {...props}
      >
        <div className="w-1 h-1 bg-current rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
        <div className="w-1 h-1 bg-current rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
        <div className="w-1 h-1 bg-current rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
      </div>
    )
  }
)
LoadingDots.displayName = 'LoadingDots'

interface LoadingSkeletonProps extends React.HTMLAttributes<HTMLDivElement> {
  width?: string | number
  height?: string | number
  variant?: 'text' | 'circular' | 'rectangular'
  animation?: 'pulse' | 'wave' | 'none'
}

const LoadingSkeleton = React.forwardRef<HTMLDivElement, LoadingSkeletonProps>(
  ({ className, width, height, variant = 'rectangular', animation = 'pulse', ...props }, ref) => {
    const variantClasses = {
      text: 'rounded',
      circular: 'rounded-full',
      rectangular: 'rounded-md',
    }

    const animationClasses = {
      pulse: 'animate-pulse',
      wave: 'shimmer',
      none: '',
    }

    const style = React.useMemo(() => {
      const computedStyle: React.CSSProperties = {}
      if (width) computedStyle.width = typeof width === 'number' ? `${width}px` : width
      if (height) computedStyle.height = typeof height === 'number' ? `${height}px` : height
      return computedStyle
    }, [width, height])

    return (
      <div
        ref={ref}
        className={cn(
          'bg-muted',
          variantClasses[variant],
          animationClasses[animation],
          className
        )}
        style={style}
        {...props}
      />
    )
  }
)
LoadingSkeleton.displayName = 'LoadingSkeleton'

interface LoadingProgressProps extends React.HTMLAttributes<HTMLDivElement> {
  value: number
  max?: number
  size?: 'sm' | 'md' | 'lg'
  variant?: 'default' | 'success' | 'warning' | 'error'
  showLabel?: boolean
  label?: string
}

const LoadingProgress = React.forwardRef<HTMLDivElement, LoadingProgressProps>(
  ({ className, value, max = 100, size = 'md', variant = 'default', showLabel = false, label, ...props }, ref) => {
  const percentage = Math.min(Math.max((value / max) * 100, 0), 100)

  const sizeClasses = {
    sm: 'h-1',
    md: 'h-2',
    lg: 'h-3',
  }

  const variantClasses = {
    default: 'bg-primary',
    success: 'bg-green-500',
    warning: 'bg-yellow-500',
    error: 'bg-red-500',
  }

  return (
    <div ref={ref} className={cn('space-y-2', className)} {...props}>
      {(showLabel || label) && (
        <div className="flex items-center justify-between text-sm">
          <span>{label || 'Progress'}</span>
          <span>{Math.round(percentage)}%</span>
        </div>
      )}
      <div className={cn('w-full bg-muted rounded-full overflow-hidden', sizeClasses[size])}>
        <div
          className={cn('h-full transition-all duration-300 ease-out', variantClasses[variant])}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  )
}
LoadingProgress.displayName = 'LoadingProgress'

interface LoadingScreenProps extends React.HTMLAttributes<HTMLDivElement> {
  message?: string
  showLogo?: boolean
}

const LoadingScreen = React.forwardRef<HTMLDivElement, LoadingScreenProps>(
  ({ className, message = 'Loading...', showLogo = true, ...props }, ref) => {
    return (
      <div
        ref={ref}
        className={cn(
          'fixed inset-0 z-50 flex items-center justify-center bg-background/80 backdrop-blur-sm',
          className
        )}
        {...props}
      >
        <div className="flex flex-col items-center space-y-4">
          {showLogo && (
            <div className="w-16 h-16 bg-gradient-to-br from-primary to-primary/60 rounded-xl flex items-center justify-center">
              <span className="text-2xl font-bold text-primary-foreground">K</span>
            </div>
          )}
          <LoadingSpinner size="lg" variant="primary" />
          <p className="text-muted-foreground animate-pulse">{message}</p>
        </div>
      </div>
    )
  }
)
LoadingScreen.displayName = 'LoadingScreen'

export {
  LoadingSpinner,
  LoadingDots,
  LoadingSkeleton,
  LoadingProgress,
  LoadingScreen,
}