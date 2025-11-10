import React from 'react';
import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from '../../lib/utils';

// Manufacturing-specific button variants
const buttonVariants = cva(
  'inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50',
  {
    variants: {
      variant: {
        default: 'bg-primary text-primary-foreground hover:bg-primary/90',
        destructive: 'bg-destructive text-destructive-foreground hover:bg-destructive/90',
        outline: 'border border-input bg-background hover:bg-accent hover:text-accent-foreground',
        secondary: 'bg-secondary text-secondary-foreground hover:bg-secondary/80',
        ghost: 'hover:bg-accent hover:text-accent-foreground',
        link: 'text-primary underline-offset-4 hover:underline',

        // Manufacturing-specific variants
        safety: 'bg-safety-red text-white hover:bg-safety-red/90 shadow-sm',
        quality: 'bg-quality-excellent text-white hover:bg-quality-excellent/90 shadow-sm',
        equipment: 'bg-status-operational text-white hover:bg-status-operational/90 shadow-sm',
        maintenance: 'bg-status-maintenance text-white hover:bg-status-maintenance/90 shadow-sm',

        // LobeChat-inspired variants
        chat: 'bg-gradient-to-r from-blue-500 to-purple-600 text-white hover:from-blue-600 hover:to-purple-700',
        quickAction: 'bg-gradient-to-r from-primary to-primary/80 text-primary-foreground hover:from-primary/90 hover:to-primary/70 shadow-sm',
      },
      size: {
        default: 'h-10 px-4 py-2',
        sm: 'h-9 rounded-md px-3',
        lg: 'h-11 rounded-md px-8',
        icon: 'h-10 w-10',
        xs: 'h-7 rounded px-2 text-xs',
      },
    },
    defaultVariants: {
      variant: 'default',
      size: 'default',
    },
  }
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  loading?: boolean;
  icon?: React.ReactNode;
  iconPosition?: 'left' | 'right';
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, loading, icon, iconPosition = 'left', children, disabled, ...props }, ref) => {
    const isDisabled = disabled || loading;

    return (
      <button
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        disabled={isDisabled}
        {...props}
      >
        {loading && (
          <svg
            className="mr-2 h-4 w-4 animate-spin"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
          >
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
            ></circle>
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
            ></path>
          </svg>
        )}

        {icon && iconPosition === 'left' && !loading && (
          <span className="mr-2">{icon}</span>
        )}

        {children}

        {icon && iconPosition === 'right' && (
          <span className="ml-2">{icon}</span>
        )}
      </button>
    );
  }
);

Button.displayName = 'Button';

// Manufacturing-specific button components
export const SafetyButton: React.FC<Omit<ButtonProps, 'variant'>> = (props) => (
  <Button variant="safety" {...props} />
);

export const QualityButton: React.FC<Omit<ButtonProps, 'variant'>> = (props) => (
  <Button variant="quality" {...props} />
);

export const EquipmentButton: React.FC<Omit<ButtonProps, 'variant'>> = (props) => (
  <Button variant="equipment" {...props} />
);

export const MaintenanceButton: React.FC<Omit<ButtonProps, 'variant'>> = (props) => (
  <Button variant="maintenance" {...props} />
);

export const QuickActionButton: React.FC<Omit<ButtonProps, 'variant'>> = (props) => (
  <Button variant="quickAction" size="sm" {...props} />
);

export { Button };
export default Button;